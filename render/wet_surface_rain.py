"""Rain-wet surface model.

The module is a small, dependency-free reference implementation of the
effects that a wet-ground renderer needs: puddle coverage, reflection
weighting, rain-driven ripples, vehicle interaction, evaporation and
night-time attenuation.

Design rules that keep the individual pieces consistent:

* Every spatial function is deterministic.  Puddles, raindrop landings and
  ripples all derive from the same value-noise / lattice hashes, so the
  particle system and the ground shading agree on where a drop lands.
* A puddle mask is layered (thin film, standing water) instead of a single
  hard threshold.  ``edge_gradient`` turns the coverage into a soft border.
* Reflection energy is split by the puddle mask: sharp screen-space
  reflections are used inside puddles, the cheaper reflection probe is used
  for the rough wet film, and dry ground gets neither.
* Rain intensity is a stateful, smoothed value.  It drives ripple speed and
  emission rate, so light and heavy rain no longer animate at one fixed rate.
* Reflection is attenuated by the scene's light level, which keeps wet
  ground from glowing at night.

All functions accept plain numbers and return plain numbers (or a small
dict when several values belong together), making the model easy to unit
test and to port to a shader.
"""

from __future__ import annotations

import math

# Seed for the shared raindrop lattice.  ``drop_landing`` and
# ``ripple_normal`` must use the same value, otherwise particles and the
# ripples on the ground would drift apart.
DROP_SEED = 90_210
_LATTICE_COLS = 64


# ---------------------------------------------------------------------------
# Small math helpers
# ---------------------------------------------------------------------------

def _clamp01(value: float) -> float:
    value = float(value)
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge0 == edge1:
        return 0.0 if x < edge0 else 1.0
    t = _clamp01((x - edge0) / (edge1 - edge0))
    return t * t * (3.0 - 2.0 * t)


def _fade(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def _hash01(ix: int, iy: int, seed: int) -> float:
    """Deterministic integer hash mapped to [0, 1]."""
    h = (int(ix) * 374_761_393 + int(iy) * 668_265_263
         + int(seed) * 2_147_483_647) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1_274_126_177) & 0xFFFFFFFF
    h ^= h >> 16
    return (h & 0xFFFFFF) / 16_777_215.0


def _value_noise(x: float, y: float, seed: int = 0) -> float:
    ix = math.floor(x)
    iy = math.floor(y)
    fx = x - ix
    fy = y - iy
    u = _fade(fx)
    v = _fade(fy)

    a = _hash01(ix, iy, seed)
    b = _hash01(ix + 1, iy, seed)
    c = _hash01(ix, iy + 1, seed)
    d = _hash01(ix + 1, iy + 1, seed)

    return _lerp(_lerp(a, b, u), _lerp(c, d, u), v)


def _fbm(x: float, y: float, seed: int = 0, octaves: int = 4) -> float:
    total = 0.0
    amplitude = 1.0
    norm = 0.0
    for octave in range(max(1, octaves)):
        total += amplitude * _value_noise(x, y, seed + octave * 131)
        norm += amplitude
        x *= 2.03
        y *= 2.03
        amplitude *= 0.5
    return total / norm if norm else 0.0


# ---------------------------------------------------------------------------
# Puddle mask and its soft edge
# ---------------------------------------------------------------------------

def puddle_mask(
    x: float,
    z: float,
    *,
    scale: float = 0.15,
    seed: int = 1337,
    layer: str = "combined",
) -> float:
    """Layered puddle coverage at world position ``(x, z)``.

    The mask is built from three noise bands so that water is not spread
    uniformly over every road-side pixel:

    * ``basin``   -- low-frequency terrain depression; water collects in
      dips instead of everywhere along the road.
    * ``cluster`` -- medium frequency groups puddles into patches.
    * ``detail``  -- high frequency breaks up the silhouette.

    Two layers are derived from the same field:

    * ``film``   -- broad, low-amplitude wet sheen (thin water).
    * ``puddle`` -- localized, full-strength standing water.

    ``layer="combined"`` (default) returns the union of both, so a puddle
    core is mirrored while its surroundings fade through a shallow film.
    Pass ``"film"`` or ``"puddle"`` to address one layer explicitly.
    """
    px = float(x) * float(scale)
    pz = float(z) * float(scale)

    basin = _fbm(px * 0.55, pz * 0.55, seed, 3)
    cluster = _fbm(px * 2.20 + 13.7, pz * 2.20 - 8.1, seed + 17, 3)
    detail = _value_noise(px * 9.0, pz * 9.0, seed + 41)

    field = 0.52 * basin + 0.33 * cluster + 0.15 * detail

    film = 0.35 * _smoothstep(0.38, 0.58, field)
    puddle = _smoothstep(0.55, 0.72, field)

    if layer == "film":
        return film
    if layer == "puddle":
        return puddle
    if layer == "combined":
        return max(film, puddle)
    raise ValueError(f"unknown puddle layer: {layer!r}")


def edge_gradient(mask: float, width: float = 0.15, *, center: float = 0.5) -> float:
    """Soften a hard-thresholded mask into a wet-to-dry ramp.

    ``width`` is the size of the transition band around ``center``.  A mask
    value well below the band is dry (0.0), well above it is fully wet (1.0)
    and values inside the band ramp smoothly.  This removes the hard puddle
    cut-out and gives the drying rim a gradual coverage falloff.
    """
    w = max(abs(float(width)), 1e-4)
    return _smoothstep(center - 0.5 * w, center + 0.5 * w, _clamp01(mask))


def puddle_depth(mask: float, max_depth: float = 0.03) -> float:
    """Water depth in metres for a puddle mask value.

    Depth grows non-linearly: a barely-covered pixel is a very shallow film,
    while the centre of a puddle approaches ``max_depth``.  Depth is useful
    for edge shading and for how long a puddle survives evaporation.
    """
    m = _clamp01(mask)
    return max(0.0, float(max_depth)) * (m ** 1.5)


# ---------------------------------------------------------------------------
# Wetness, roughness and reflection
# ---------------------------------------------------------------------------

def wet_roughness(
    wetness: float,
    mask: float = 0.0,
    *,
    dry: float = 0.85,
    wet_film: float = 0.28,
    puddle: float = 0.02,
) -> float:
    """Surface roughness for a given wetness and puddle mask.

    Drying ground first loses its mirror finish (roughness rises quickly),
    then the residual film disappears.  Inside a puddle the water surface is
    almost perfectly smooth, independent of the roughness of the asphalt
    below it.
    """
    w = _clamp01(wetness)
    film_roughness = _lerp(float(dry), float(wet_film), w)
    puddle_weight = edge_gradient(mask, 0.2)
    return _lerp(film_roughness, float(puddle), puddle_weight)


def humidity_reflect(
    humidity: float,
    cos_theta: float = 1.0,
    *,
    base: float = 0.02,
) -> float:
    """Specular reflectance of a wet surface.

    ``cos_theta`` is the cosine between the view direction and the surface
    normal (1.0 = looking straight down, 0.0 = grazing).  Humidity now
    changes reflection *strength*, not only roughness: a soaked surface has
    a thicker water film and therefore a much stronger specular response,
    while dry ground keeps the small Fresnel base value.
    """
    h = _clamp01(humidity)
    cos_t = _clamp01(abs(float(cos_theta)))
    base = _clamp01(base)

    # Schlick approximation for the underlying surface.
    fresnel = base + (1.0 - base) * (1.0 - cos_t) ** 5

    # Water film thickens with humidity; at normal incidence the wet surface
    # still gains a broad specular lobe instead of keeping the dry value.
    film = h ** 1.5
    strength = base * (1.0 - 0.5 * h) + (1.0 - base) * film * (0.30 + 0.70 * (1.0 - cos_t))
    return _clamp01(strength + 0.02 * fresnel)


def reflect_probe(wetness: float, mask: float = 0.0) -> float:
    """Share of reflection energy taken from the reflection probe.

    The split is allocated by the puddle mask instead of a fixed ratio:

    * puddles rely on sharp screen-space reflections, so the probe share is
      small (it only contributes a little diffuse ambient);
    * the rough wet film around a puddle relies on the blurred probe;
    * dry ground receives essentially no reflection at all.

    Multiply the result by the reflection colour and mix with
    ``1 - reflect_probe`` of the SSR result.
    """
    puddle = edge_gradient(mask, 0.2)
    film = _clamp01(wetness) * (1.0 - puddle)
    return _clamp01(0.75 * film + 0.15 * puddle)


def mask_blend(ssr: float, probe: float, mask: float = 0.0, *, edge_width: float = 0.15) -> float:
    """Blend SSR and probe samples by puddle coverage.

    Dry pixels return the probe sample (and should be attenuated by
    ``reflect_probe``/humidity), puddle centres return the sharp SSR sample,
    and the puddle rim returns a smooth mixture.  ``mask`` may be the raw
    combined mask from :func:`puddle_mask`.
    """
    weight = edge_gradient(mask, edge_width)
    return _lerp(float(probe), float(ssr), weight)


def night_atten(reflectance: float, light_level: float, *, floor: float = 0.08) -> float:
    """Attenuate reflection by the scene's light level.

    Wet ground reflects the environment, but at night there is little
    environment light and the reflected image must not stay as bright as in
    daylight.  ``floor`` keeps a small amount of ambient/specular so the
    surface does not turn completely black.
    """
    r = max(0.0, float(reflectance))
    level = _clamp01(float(light_level))
    return r * (_clamp01(floor) + (1.0 - _clamp01(floor)) * level)


def scene_reflection(cos_theta: float, mask: float = 1.0, *, scene_strength: float = 0.85) -> float:
    """Fraction of the reflection that comes from the local scene capture.

    A flat sky probe alone leaves buildings and characters out of the
    reflection.  The renderer should mix the sky probe with a scene capture
    (planar/box reflection) using this weight.  At grazing angles the
    reflected ray points at nearby buildings and characters, so the local
    scene dominates; when looking almost straight down, the reflected ray
    points at the sky.  ``mask`` limits the effect to water-covered pixels.
    """
    cos_t = _clamp01(abs(float(cos_theta)))
    grazing = (1.0 - cos_t) ** 2
    return _clamp01(float(scene_strength) * _clamp01(mask) * (0.25 + 0.75 * grazing))


# ---------------------------------------------------------------------------
# Rain intensity, drops and ripples
# ---------------------------------------------------------------------------

def rain_intensity(
    rain_rate: float,
    dt: float,
    *,
    prev: float = 0.0,
    rise_time: float = 0.6,
    fall_time: float = 1.2,
) -> float:
    """Smooth rain intensity in [0, 1] from the raw rain rate.

    Rain cannot switch on or off instantly.  The value approaches the target
    rate with an exponential response, using a faster time constant when the
    rain builds up than when it eases off.  The result feeds
    :func:`ripple_normal` (scroll speed), :func:`spray_emit` and the wetness
    buildup, so all rain-driven effects share one intensity value.
    """
    target = _clamp01(rain_rate)
    previous = _clamp01(prev)
    tau = rise_time if target >= previous else fall_time
    tau = max(float(tau), 1e-4)
    step = max(float(dt), 0.0)
    k = 1.0 - math.exp(-step / tau)
    return _clamp01(previous + (target - previous) * k)


def _drop_period(intensity: float) -> float:
    """Time between successive drops in one lattice cell."""
    return 1.4 - 0.7 * _clamp01(intensity)


def _lattice_origin(ix: int, iy: int, spacing: float) -> tuple[float, float, float]:
    """Shared drop origin/phase for a lattice cell.

    Both :func:`drop_landing` and :func:`ripple_normal` go through this
    helper, which guarantees that a particle's landing point and the ripple
    centre on the ground are the same world position.
    """
    jx = _hash01(ix, iy, DROP_SEED)
    jz = _hash01(ix, iy, DROP_SEED + 101)
    phase = _hash01(ix, iy, DROP_SEED + 211)
    return (ix + jx) * spacing, (iy + jz) * spacing, phase


def drop_landing(
    drop_id: int,
    elapsed: float,
    *,
    spacing: float = 3.0,
    intensity: float = 1.0,
) -> dict:
    """Landing state for particle/drop ``drop_id`` after ``elapsed`` seconds.

    The drop belongs to the same lattice that the ground ripple shader
    samples, so the reported ``x``/``z`` is also the ripple origin.  ``age``
    is the time since the drop landed in its current cycle (0..period);
    ``alive`` says whether its ripple is still within its lifetime.
    """
    did = int(drop_id)
    ix = did % _LATTICE_COLS
    iy = (did // _LATTICE_COLS) % _LATTICE_COLS

    x, z, phase = _lattice_origin(ix, iy, float(spacing))
    period = _drop_period(intensity)
    age = (float(elapsed) + phase * period) % period
    lifetime = period * 0.9

    speed = 1.6 + 2.4 * _clamp01(intensity)
    return {
        "x": x,
        "z": z,
        "age": age,
        "period": period,
        "alive": age < lifetime,
        "radius": speed * age,
        "strength": 0.5 + 0.5 * _hash01(ix, iy, DROP_SEED + 307),
    }


def ripple_normal(
    x: float,
    z: float,
    intensity: float = 1.0,
    time: float = 0.0,
    mask: float = 1.0,
    *,
    spacing: float = 3.0,
    max_radius: float = 2.8,
) -> tuple[float, float]:
    """Surface normal perturbation from rain ripples at ``(x, z)``.

    Ripple speed is driven by rain intensity (light rain expands slowly,
    heavy rain quickly) instead of a fixed scroll rate.  Each lattice cell
    emits a drop whose origin matches :func:`drop_landing`; the expanding
    ring is a decaying gaussian bump, and its derivative is accumulated into
    an (nx, nz) normal offset.  ``mask`` restricts ripples to water-covered
    ground.
    """
    inten = _clamp01(intensity)
    coverage = _clamp01(mask)
    amplitude = 0.35 * inten * coverage
    if amplitude <= 0.0:
        return (0.0, 0.0)

    period = _drop_period(inten)
    speed = 1.6 + 2.4 * inten
    width = 0.25 + 0.15 * inten
    max_radius = max(0.1, float(max_radius))

    cx0 = math.floor(float(x) / spacing)
    cz0 = math.floor(float(z) / spacing)

    nx = 0.0
    nz = 0.0

    for cx in range(cx0 - 1, cx0 + 2):
        for cz in range(cz0 - 1, cz0 + 2):
            ox, oz, phase = _lattice_origin(cx, cz, spacing)
            age = (float(time) + phase * period) % period
            radius = speed * age
            if radius > max_radius:
                continue

            dx = float(x) - ox
            dz = float(z) - oz
            distance = math.hypot(dx, dz)
            if distance < 1e-6:
                continue

            # Gaussian ring centred on the expanding radius; its derivative
            # gives the local slope of the water surface.
            u = (distance - radius) / width
            ring = math.exp(-u * u) * math.exp(-age * 1.6)
            slope = -2.0 * u / width * ring * amplitude
            nx += slope * dx / distance
            nz += slope * dz / distance

    # Keep the perturbation in a physically plausible range.
    nx = max(-0.5, min(0.5, nx))
    nz = max(-0.5, min(0.5, nz))
    return (nx, nz)


# ---------------------------------------------------------------------------
# Vehicle interaction and evaporation
# ---------------------------------------------------------------------------

def vehicle_interaction(
    speed: float,
    rain_intensity: float = 1.0,
    *,
    mask: float = 0.5,
    wheel_contact: float = 1.0,
) -> dict:
    """Spray and tyre-track response for a vehicle crossing wet ground.

    Fast vehicles over standing water throw a sheet of spray and press water
    into the tyre grooves, leaving a wet track behind.  Both effects scale
    with speed, puddle coverage and rain intensity, so a car on a dry road
    produces nothing.  ``spray_rate`` is particles/second for one wheel;
    ``trail_wetness`` is the additional wetness left in the tyre track and
    ``wake`` is the outward water displacement used to fade the track.
    """
    v = max(0.0, float(speed))
    wet = _clamp01(rain_intensity)
    puddle = _clamp01(mask)
    contact = _clamp01(wheel_contact)

    water = puddle * wet
    speed_term = v / (v + 7.0)

    spray = _clamp01(water * speed_term * 1.4 * contact)
    trail = _clamp01(0.45 * water + 0.25 * min(v / 12.0, 1.0))
    wake = _clamp01(0.6 * water * min(v / 10.0, 1.0))

    return {
        "spray": spray,
        "spray_rate": spray_emit(v, water, wheel_contact=contact),
        "trail_wetness": trail,
        "wake": wake,
        "track_mask": trail * (1.0 - wake),
    }


def spray_emit(
    speed: float,
    intensity: float = 1.0,
    *,
    wheel_contact: float = 1.0,
    max_rate: float = 240.0,
) -> float:
    """Spray particle emission rate (particles/second per wheel).

    Speed and standing water both matter: a stationary car does not spray,
    and a fast car on dry asphalt does not either.
    """
    v = max(0.0, float(speed))
    wet = _clamp01(intensity)
    contact = _clamp01(wheel_contact)
    return max(0.0, float(max_rate)) * wet * contact * (v / (v + 7.0))


def evaporate(
    wetness: float,
    dt: float,
    *,
    rate: float = 0.35,
    airflow: float = 0.0,
    heat: float = 0.0,
) -> float:
    """Advance wetness by ``dt`` seconds of drying.

    When the rain stops the ground does not become dry in one frame.  Thin
    film dries first; deep puddles hold water much longer.  ``airflow`` and
    ``heat`` speed up evaporation.  The returned wetness can be fed back
    into :func:`wet_roughness`, :func:`humidity_reflect` and
    :func:`puddle_mask`-driven shading during the transition.
    """
    w = _clamp01(wetness)
    if w <= 0.0:
        return 0.0
    step = max(0.0, float(dt))
    k = max(0.0, float(rate)) * (1.0 + 0.8 * _clamp01(airflow) + 0.6 * _clamp01(heat))

    # Thin water film (low w) has a shorter lifetime than a deep puddle.
    lifetime_scale = 1.2 - 0.5 * w
    return _clamp01(w * math.exp(-k * step * lifetime_scale))
