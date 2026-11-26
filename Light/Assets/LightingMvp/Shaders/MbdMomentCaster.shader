Shader "Hidden/LightingMvp/MbdMomentCaster"
{
    Properties
    {
        _BaseMap ("Base Map", 2D) = "white" {}
        _BaseColor ("Base Color", Color) = (1, 1, 1, 1)
        _Cutoff ("Cutoff", Float) = 0.5
        _AlphaClip ("Alpha Clip", Float) = 0
        _Cull ("Cull", Float) = 2
    }
    SubShader
    {
        Tags { "RenderPipeline" = "UniversalPipeline" }
        Pass
        {
            Cull [_Cull] ZWrite On ZTest LEqual
            HLSLPROGRAM
            #pragma target 4.5
            #pragma vertex Vert
            #pragma fragment Frag
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            TEXTURE2D(_BaseMap);
            SAMPLER(sampler_BaseMap);
            CBUFFER_START(UnityPerMaterial)
                float4 _BaseMap_ST;
                float4 _BaseColor;
                float _Cutoff;
                float _AlphaClip;
            CBUFFER_END
            float4x4 _E20ViewProjection;
            float4 _E20LightPositionInvRange;
            struct Attributes { float3 position : POSITION; float2 uv : TEXCOORD0; };
            struct Varyings { float4 clip : SV_POSITION; float3 world : TEXCOORD0; float2 uv : TEXCOORD1; };
            Varyings Vert(Attributes input)
            {
                Varyings o;
                o.world = TransformObjectToWorld(input.position);
                o.clip = mul(_E20ViewProjection, float4(o.world, 1));
                o.uv = TRANSFORM_TEX(input.uv, _BaseMap);
                return o;
            }
            float4 Frag(Varyings input) : SV_Target
            {
                if (_AlphaClip > 0.5)
                    clip(SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, input.uv).a * _BaseColor.a - _Cutoff);
                float depth = 2.0 * saturate(length(input.world - _E20LightPositionInvRange.xyz) * _E20LightPositionInvRange.w) - 1.0;
                float squared = depth * depth;
                return float4(depth, squared, squared * depth, squared * squared);
            }
            ENDHLSL
        }
    }
}
