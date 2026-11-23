# Unity Cornell Box lighting-optimization MVP

The Unity MVP recovers a fixed rig of Point and Spot lights from synthetic, world-space direct irradiance or total diffuse outgoing radiance. The saved `Assets/Scenes/CornellBoxMvp.unity` scene contains an open-front room, two Cornell-style rotated blocks with box colliders, a validation camera, and three candidate light slots. The scene is a local Unity asset and is not included in the source-only Git checkout.

## Run and inspect

1. Open the `Light/` project in Unity 2022.3.62f3c1 with the saved Cornell scene available.
2. Run **Lighting MVP > Run Cornell Box Experiments**. The runner reads the saved scene, temporarily applies each target/initial/optimized rig for previews, and restores the scene lights. It does not save the scene.
3. In **Lighting MVP > Debug Window > Iteration Replay**, select a **Three points** experiment (a multi-light recording is preferred by default). Press **Play** or drag **Recorded state**. With **Replay view > All lights**, every light follows the same recorded state. To inspect one light, choose **Selected light only**, then select **Light 1**, **Light 2**, or **Light 3**. Only that replay light, its Scene markers, and its parameters are shown. Switching views or lights keeps the current state; loss and field NRMSE still describe the full rig. Use **Restore scene lights** to end the preview.
4. Turn on **Show Irradiance** for a direct-light heatmap. For **Three points: multi-bounce GI + CCT**, click **Show multi-bounce field** to see the actual solver patches in the Scene view. **Advanced Irradiance Settings** can show the current field, target field, or absolute error and export it as CSV. In single-light replay these fields use only the selected light and its matching target. The Game camera renders the editable Unity lights; the multi-bounce solver does not inject its field into Unity's render pipeline.
5. Use **Reload recorded runs** if the window was already open when the reports were regenerated. The three-light runs use the same deterministic position offsets (about 0.38–0.44 m) and intensity multipliers (0.55×, 1.55×, 0.65×). The color-optimization variants also start at 4200/3300/5400 K instead of the target 2700/5000/8000 K. Jump between the first and last states or lower **States / second** for a slower comparison.

**Light Perturbation** works on Point, Spot, and Area lights in the current scene. Edit the bounding box in the Scene view, set the seed and perturbation scales, then apply the change. The same seed and starting rig reproduce the same perturbation; the panel supports Undo and restoring the starting rig.

The batch command requires a graphics device for Compute Shader and camera capture. Close the project in Unity first, or run against a separate validation copy of the project:

```powershell
& 'E:\UnityEngine\2022.3.62f3c1\Editor\Unity.exe' -batchmode -projectPath 'E:\Light\Light' -executeMethod LightingMvp.Editor.CornellBoxMvpEditor.BatchRun -logFile 'E:\Light\results\unity_mvp_editor.log'
```

## Lighting field and gradients

- `SurfaceSampleSet` reads world-space receiver positions, normals, diffuse material colors, and area weights from the saved meshes. It samples the room quads and subdivides the block triangles.
- `UnityVisibility` raycasts between each light and surface receiver against Unity colliders. The mask is recomputed for each candidate rig and held fixed when computing gradients. Visibility jumps across shadow boundaries can still affect convergence.
- `AnalyticLighting` sums direct irradiance from the rig and provides explicit C# gradients for position, log intensity, Spot direction, and RGB or correlated color temperature (CCT). `LightRigOptimizer` runs Adam and L-BFGS against the area-weighted target field.
- `SurfaceLightingGpu` evaluates direct irradiance and direct Lambertian radiance with the same light-to-surface mask. It does not differentiate the Unity render pipeline.
- `DiffuseRadiosity` ports the research line's fixed diffuse transport solve to the Unity mesh patches. It estimates reciprocal form factors with Unity collider rays and 2×2 or 4×4 quadrature on close triangle/quad patches, then bounds residual energy overestimates symmetrically. The CPU factors `I - diag(albedo) F` once per channel, solves all bounces for each new rig, and uses the transpose solve to propagate the loss gradient to direct irradiance. Unity uses its own patch geometry and colliders; this is not a pixel-for-pixel Mitsuba comparison or textured/specular GI.

## Experiments and outputs

| Experiment | Optimized parameters | Visibility |
|---|---|---|
| `point_position_intensity` | Point position, intensity | Off (regression baseline) |
| `spot_position_direction` | Spot position, direction | Off (regression baseline) |
| `three_point_position_intensity` | Three positions and intensities, with fixed 2700 K / 5000 K / 8000 K | Unity collider rays |
| `three_point_joint_cct` | Three positions, intensities, and CCTs | Unity collider rays |
| `three_point_radiosity_cct` | Same three lights and perturbation, fitted to total multi-bounce diffuse radiance | Unity light and patch visibility rays |

`results/unity_mvp/` contains `report.json`, per-experiment JSON and CSV histories, irradiance field CSVs, GPU irradiance/radiance CSVs, and target/initial/optimized PNGs. The report separates field NRMSE from per-light recovery: it lists both label-matched and position-aligned errors, along with ray-visibility mismatch. Validation checks CPU/GPU agreement, verifies that shadows materially change the target field, and compares C# gradients with central differences at a fixed visibility mask. A separate flag records whether the actual mask changes under those perturbations. Each recovery must reach field NRMSE below 2%. After matching by position, per-light position, intensity, direction, and CCT errors must be below 2 cm, 5%, 2°, and 200 K respectively, where applicable; the ray-mask mismatch must be below 1% for shadowed cases. Batch validation also exercises multi-light replay and the radiance overlay.

The radiosity report compares 0, 1, 4, and 16 bounces with the complete linear solve and checks its fixed-point residual and fixed-visibility gradients. The field CSVs contain the fitted quantity for their experiment. These are same-model recovery experiments from known starts; imported targets and recovery from arbitrary starts are separate tests.
