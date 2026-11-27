Shader "Hidden/LightingMvp/MbdMomentCaster"
{
    Properties
    {
        _BaseMap ("Base Map", 2D) = "white" {}
        _BaseColor ("Base Color", Color) = (1, 1, 1, 1)
        _Cutoff ("Cutoff", Float) = 0.5
        _AlphaClip ("Alpha Clip", Float) = 0
        _Cull ("Cull", Float) = 2
        _SignedMoments ("Signed depth moments", Float) = 1
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
                float _SignedMoments;
            CBUFFER_END
            float4x4 _E20ViewProjection;
            float4 _E20LightPositionInvRange;
            float _E22CasterDepthBias;
            struct Attributes { float3 position : POSITION; float2 uv : TEXCOORD0; };
            struct Varyings { float4 clip : SV_POSITION; float3 world : TEXCOORD0; float2 uv : TEXCOORD1; };
            Varyings Vert(Attributes input)
            {
                Varyings o;
                o.world = TransformObjectToWorld(input.position);
                if (_E22CasterDepthBias != 0)
                    o.world += _E22CasterDepthBias * normalize(_E20LightPositionInvRange.xyz - o.world);
                o.clip = mul(_E20ViewProjection, float4(o.world, 1));
                o.uv = TRANSFORM_TEX(input.uv, _BaseMap);
                return o;
            }
            float4 Frag(Varyings input) : SV_Target
            {
                if (_AlphaClip > 0.5)
                    clip(SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, input.uv).a * _BaseColor.a - _Cutoff);
                float depth = saturate(length(input.world - _E20LightPositionInvRange.xyz) * _E20LightPositionInvRange.w);
                if (_SignedMoments > 0.5) depth = 2.0 * depth - 1.0;
                float squared = depth * depth;
                return float4(depth, squared, squared * depth, squared * squared);
            }
            ENDHLSL
        }
    }
}
