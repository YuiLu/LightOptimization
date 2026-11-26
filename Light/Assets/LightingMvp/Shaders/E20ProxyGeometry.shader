Shader "Hidden/LightingMvp/E20ProxyGeometry"
{
    Properties
    {
        _BaseColor ("Base Color", Color) = (1, 1, 1, 1)
        _Metallic ("Metallic", Float) = 0
        _Cull ("Cull", Float) = 2
    }
    SubShader
    {
        Tags { "RenderPipeline" = "UniversalPipeline" }
        Cull [_Cull]
        ZWrite On
        ZTest LEqual

        HLSLINCLUDE
        #pragma target 4.5
        #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

        CBUFFER_START(UnityPerMaterial)
        float4 _BaseColor;
        float _Metallic;
        CBUFFER_END

        float4x4 _E20ViewProjection;
        float4 _E20LightPositionInvRange;

        struct Attributes
        {
            float3 position : POSITION;
            float3 normal : NORMAL;
        };

        struct Varyings
        {
            float4 clip : SV_POSITION;
            float3 position : TEXCOORD0;
            float3 normal : TEXCOORD1;
        };

        Varyings Vert(Attributes input)
        {
            Varyings output;
            output.position = TransformObjectToWorld(input.position);
            output.normal = TransformObjectToWorldNormal(input.normal);
            output.clip = mul(_E20ViewProjection, float4(output.position, 1.0));
            return output;
        }
        ENDHLSL

        Pass
        {
            Name "FixedReceiverAttributes"
            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment AttributesFragment

            struct Receiver
            {
                float4 position : SV_Target0;
                float4 normal : SV_Target1;
                float4 diffuse : SV_Target2;
            };
            Receiver AttributesFragment(Varyings input)
            {
                Receiver o;
                o.position = float4(input.position, 1);
                o.normal = float4(normalize(input.normal), 1);
                o.diffuse = float4(_BaseColor.rgb * (0.96 * (1.0 - _Metallic)), 1);
                return o;
            }
            ENDHLSL
        }

        Pass
        {
            Name "RadialDepthMoments"
            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment MomentsFragment
            float4 MomentsFragment(Varyings input) : SV_Target
            {
                float depth = saturate(length(input.position - _E20LightPositionInvRange.xyz) * _E20LightPositionInvRange.w);
                float square = depth * depth;
                return float4(depth, square, square * depth, square * square);
            }
            ENDHLSL
        }
    }
}
