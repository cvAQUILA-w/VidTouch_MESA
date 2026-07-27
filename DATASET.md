# Dataset Interface

## Evidence Unit

MESA treats a physical Fabric ID as the learning unit. RGB images and tactile
videos are independent observations rather than synchronized frame pairs. At
training time, FPR can recombine any valid RGB view and tactile trial belonging
to the same Fabric ID.

## Labels

The four target axes are:

- `weave`: single-label textile construction;
- `material`: single-label ordered material composition;
- `usage`: single-label intended use;
- `features`: multi-label functional or perceptual properties.

Material strings preserve supplier-provided component order. Reversing a
composition string is therefore not treated as an automatic synonym.

## Closed-Set Projection

The canonical annotation release contains rare labels that are useful for
future open-set and few-shot work. The paper's closed-set benchmark retains
labels supported by at least three fabrics for weave, material, and usage and
at least five fabrics for features. This yields 10/13/11/14 output labels.

Filtering is attribute-specific:

- a fabric with an excluded material may still supervise weave, usage, and
  retained features;
- excluded labels are absent from the corresponding output head;
- the same masks are used for loss computation and scoring.

## Media Assumptions

The loader supports common image extensions and MP4/MOV/AVI/MKV video files.
Filenames must begin with the Fabric ID appearing in `label.txt`. The default
configuration samples 12 frames at 160 x 160 from each tactile video and uses
224 x 224 RGB inputs.

The tactile videos are dynamic optical-contact evidence. They are not calibrated
measurements of force, friction, stiffness, thickness, or fiber composition.
