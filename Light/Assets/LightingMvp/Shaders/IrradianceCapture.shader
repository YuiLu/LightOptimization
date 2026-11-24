Shader "Hidden/LightingMvp/IrradianceCapture"
{
    Properties
    {
        [Enum(UnityEngine.Rendering.CullMode)] _Cull ("Cull", Float) = 2
    }
    SubShader
    {
        Tags { "RenderPipeline" = "UniversalPipeline" }
        Pass
        {
            Name "NativeDirectIrradiance"
            Cull [_Cull] ZWrite On ZTest LEqual Blend Off

            HLSLPROGRAM
            #pragma target 3.5
            #pragma vertex Vert
            #pragma fragment Frag
            #pragma multi_compile_instancing
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS _MAIN_LIGHT_SHADOWS_CASCADE _MAIN_LIGHT_SHADOWS_SCREEN
            #pragma multi_compile _ _ADDITIONAL_LIGHTS_VERTEX _ADDITIONAL_LIGHTS
            #pragma multi_compile_fragment _ _ADDITIONAL_LIGHT_SHADOWS
            #pragma multi_compile_fragment _ _SHADOWS_SOFT
            #pragma multi_compile_fragment _ _LIGHT_COOKIES
            #pragma multi_compile_fragment _ _LIGHT_LAYERS
            #pragma multi_compile _ _FORWARD_PLUS
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"

            struct Attributes
            {
                float4 positionOS : POSITION;
                float3 normalOS : NORMAL;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };
            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float3 positionWS : TEXCOORD0;
                float3 normalWS : TEXCOORD1;
                float4 shadowCoord : TEXCOORD2;
                float3 vertexLighting : TEXCOORD3;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            Varyings Vert(Attributes input)
            {
                Varyings output = (Varyings)0;
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_TRANSFER_INSTANCE_ID(input, output);
                VertexPositionInputs position = GetVertexPositionInputs(input.positionOS.xyz);
                output.positionCS = position.positionCS;
                output.positionWS = position.positionWS;
                output.normalWS = TransformObjectToWorldNormal(input.normalOS);
                #if defined(_MAIN_LIGHT_SHADOWS_SCREEN)
                    output.shadowCoord = GetShadowCoord(position);
                #endif
                #if defined(_ADDITIONAL_LIGHTS_VERTEX)
                    output.vertexLighting = VertexLighting(output.positionWS, output.normalWS);
                #endif
                return output;
            }

            float3 Incident(Light light, float3 normalWS, uint meshLayers)
            {
                #if defined(_LIGHT_LAYERS)
                    if (!IsMatchingLightLayer(light.layerMask, meshLayers)) return 0;
                #endif
                return float3(light.color) * (light.distanceAttenuation * light.shadowAttenuation * saturate(dot(normalWS, light.direction)));
            }

            float4 Frag(Varyings input, FRONT_FACE_TYPE face : FRONT_FACE_SEMANTIC) : SV_Target
            {
                UNITY_SETUP_INSTANCE_ID(input);
                InputData inputData = (InputData)0;
                inputData.positionWS = input.positionWS;
                inputData.normalizedScreenSpaceUV = GetNormalizedScreenSpaceUV(input.positionCS);
                float3 normalWS = NormalizeNormalPerPixel(input.normalWS) * IS_FRONT_VFACE(face, 1.0, -1.0);
                uint meshLayers = GetMeshRenderingLayer();
                const half4 noBakedMask = half4(1, 1, 1, 1);
                #if defined(_MAIN_LIGHT_SHADOWS_SCREEN)
                    float4 shadowCoord = input.shadowCoord;
                #else
                    float4 shadowCoord = TransformWorldToShadowCoord(input.positionWS);
                #endif
                float3 irradiance = Incident(GetMainLight(shadowCoord, input.positionWS, noBakedMask), normalWS, meshLayers);
                #if defined(_ADDITIONAL_LIGHTS)
                    #if USE_FORWARD_PLUS
                        for (uint lightIndex = 0; lightIndex < min(URP_FP_DIRECTIONAL_LIGHTS_COUNT, MAX_VISIBLE_LIGHTS); ++lightIndex)
                            irradiance += Incident(GetAdditionalLight(lightIndex, input.positionWS, noBakedMask), normalWS, meshLayers);
                    #endif
                    uint lightCount = GetAdditionalLightsCount();
                    LIGHT_LOOP_BEGIN(lightCount)
                        irradiance += Incident(GetAdditionalLight(lightIndex, input.positionWS, noBakedMask), normalWS, meshLayers);
                    LIGHT_LOOP_END
                #elif defined(_ADDITIONAL_LIGHTS_VERTEX)
                    irradiance += input.vertexLighting;
                #endif
                // RGB is linear incident direct light. Alpha is positive linear eye depth in metres; 0 = no receiver.
                return float4(irradiance, -TransformWorldToView(input.positionWS).z);
            }
            ENDHLSL
        }
    }
}
