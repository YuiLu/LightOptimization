# Unity Cornell Box lighting-optimization MVP

This is the Unity counterpart of `mvp/lighting.py` and the LightOpt-style experiments in `mvp/run_experiments.py`.

## Scene asset

`Assets/Scenes/CornellBoxMvp.unity` is the saved experimental scene asset. There is no procedural Cornell Box builder. `SurfaceSampleSet.FromRenderers` reads its world-space mesh vertices, normals, area weights, and `_BaseColor` materials. The current lightweight scene adapter expects quad surfaces; a general triangle/Poisson sampler can replace it without changing the optimizer.

## Run and inspect every iteration

1. Open `CornellBoxMvp.unity` in Unity 2022.3.
2. Choose **Lighting MVP > Run Cornell Box Experiments**. This validates the implementation, runs both experiments, and writes reports, CSV fields/histories, and PNG previews to `../results/unity_mvp`.
3. Choose **Lighting MVP > Debug Window** and open **Iteration Replay**. Select Point or Spot, drag the recorded-state slider, or press Play. The Game view shows the changing Unity light; Scene-view markers show the current and target positions.

The **Show Irradiance** toggle (with an optional advanced-settings foldout) evaluates the saved Cornell Box surfaces with the Compute Shader and draws continuous, vertex-colored surface heatmaps in both Scene and Game views. Irradiance is evaluated on a dense surface grid and smoothly interpolated across each mesh; the transient overlay meshes are never saved into the scene asset. Discrete sample discs remain available as an optional debugging mode. It supports current scene lights, Point/Spot target fields, and absolute current-versus-target error. Display controls include sampling density, RGB or heatmap coloring (Viridis or Blue-Cyan-Green-Yellow-Red), automatic/manual normalization, logarithmic brightness, marker size, automatic refresh when lights change, and CSV export of the displayed field.

The **Light Perturbation** tab operates on the current scene, not just Cornell Box. An editable world-space Bounding Box selects Point, Spot, and Area lights by position. The box is edited directly in the Scene view: drag the center transform handle to move it and the six face controls to resize it. Numeric center/size fields are intentionally omitted. On first opening the perturbation tab, the box automatically fits the current scene lights. Controls include random seed, position sigma, log-intensity sigma, direction sigma, optional linear-color sigma, selected-object filtering, fitting the box to lights, selecting in-scope lights, Undo, and restoration of the pre-perturbation state. Applying a perturbation increments the seed; the same seed and starting state reproduce the same perturbation.

Every Adam step and every accepted L-BFGS step records the complete light specification, loss, and field NRMSE. The history CSV also contains position, intensity, and direction, so the trajectory can be plotted or exported without Unity.

For automation:

```powershell
E:\UnityEngine\2022.3.62f3c1\Editor\Unity.exe -batchmode -projectPath E:\Light\Light -executeMethod LightingMvp.Editor.CornellBoxMvpEditor.BatchRun -logFile E:\Light\results\unity_mvp_editor.log
```

## Architecture and differentiability

- `SurfaceSampleSet`: camera-free surface positions, normals, area weights, names, and diffuse albedos read from scene assets.
- `SurfaceLightingGpu` + `SurfaceLighting.compute`: evaluates world-space direct irradiance `E` and Lambertian outgoing radiance `L_o = albedo * E / pi`, then reads both back to C#. This is not a screen-color approximation.
- `AnalyticLighting`: C# point/spot attenuation equations and explicit C# analytic derivatives for position, log intensity, raw direction, and color. It mirrors the Python equations.
- `LightRigOptimizer`: renderer-independent C# Adam, normalized L-BFGS refinement, parameter-group learning rates, and per-state optimization history.
- `UnityLightAdapter`: converts optimized parameters to editable Unity Light components.
- `CornellBoxMvpEditor`: loads the experiment scene, generates target fields, runs GT recovery, and exports results.
- `LightingDebugWindow`: replays optimization trajectories and creates reproducible perturbations of existing scene lights.

The differentiable pipeline is specifically:

```text
Saved scene surface samples
        -> analytic direct-irradiance forward model (CPU / Compute Shader)
        -> explicit C# analytic parameter gradients
        -> Adam + L-BFGS
        -> Unity Light parameters
```

It is **not** a fully differentiable Unity/URP rendering pipeline. The Compute Shader is a forward backend, not an autodiff system. URP camera images, shadow visibility, and multiple-bounce GI do not participate in backpropagation. Current experiments use direct lighting with visibility = 1 and no internal occluders.

No native C++ rendering plugin is needed at this stage. A custom SRP pass or native plugin becomes relevant when extracting Unity's actual rendered lighting/G-buffer data, but those screen-space buffers are not a substitute for this camera-independent surface field.

## Validation and next steps

The automated run checks Point and Spot CPU/GPU agreement and analytic gradients against central finite differences. A run succeeds when both lighting fields reach NRMSE < 2%.

Next backend steps are Unity ray-query visibility for candidate lights, internal occluders, target-field import, and indirect lighting if required. Hard shadow visibility is not differentiable and should use detached visibility or an explicitly smoothed approximation.






