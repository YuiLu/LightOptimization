Shader "LightingMvp/Stylized Point Response"
{
    Properties
    {
        _BaseColor("Base Color", Color) = (0.8, 0.8, 0.8, 1)
        [HideInInspector] _BaseMap("Base Map", 2D) = "white" {}
        [HideInInspector] _Cull("Cull", Float) = 2
        [HideInInspector] _Cutoff("Cutoff", Range(0,1)) = 0.5
    }
    SubShader
    {
        Tags { "RenderPipeline"="UniversalPipeline" "RenderType"="Opaque" }
        Pass
        {
            Name "StylizedForward"
            Tags { "LightMode"="UniversalForward" }
            Cull [_Cull]
            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag
            #pragma multi_compile _ _ADDITIONAL_LIGHTS
            #pragma multi_compile_fragment _ _ADDITIONAL_LIGHT_SHADOWS
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"

            CBUFFER_START(UnityPerMaterial)
                float4 _BaseColor;
            CBUFFER_END

            struct Attributes { float4 positionOS : POSITION; float3 normalOS : NORMAL; };
            struct Varyings { float4 positionCS : SV_POSITION; float3 positionWS : TEXCOORD0; float3 normalWS : TEXCOORD1; };

            Varyings Vert(Attributes input)
            {
                VertexPositionInputs p = GetVertexPositionInputs(input.positionOS.xyz);
                Varyings o;
                o.positionCS = p.positionCS;
                o.positionWS = p.positionWS;
                o.normalWS = TransformObjectToWorldNormal(input.normalOS);
                return o;
            }

            float4 Frag(Varyings input) : SV_Target
            {
                float3 n = normalize(input.normalWS);
                float3 v = GetWorldSpaceNormalizeViewDir(input.positionWS);
                float3 rgb = 0.006 * _BaseColor.rgb;
                uint count = GetAdditionalLightsCount();
                for (uint i = 0; i < count; ++i)
                {
                    Light light = GetAdditionalLight(i, input.positionWS, half4(1, 1, 1, 1));
                    float ndl = saturate(dot(n, light.direction));
                    // Artistic bands deliberately have hard edges; the solver never sees this function.
                    float ramp = (0.28 * step(0.18, ndl) + 0.38 * step(0.48, ndl) + 0.48 * step(0.76, ndl));
                    float rim = step(0.45, 1 - saturate(dot(n, v))) * step(0.40, ndl);
                    rgb += light.color * light.distanceAttenuation * light.shadowAttenuation
                        * (_BaseColor.rgb * (ramp + 0.15 * ndl) + float3(0.09, 0.21, 0.39) * rim);
                }
                return float4(rgb, 1);
            }
            ENDHLSL
        }
        UsePass "Universal Render Pipeline/Lit/ShadowCaster"
    }
}
