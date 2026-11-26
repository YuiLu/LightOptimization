#ifndef LIGHTING_MVP_MOMENT_BOUNDS
#define LIGHTING_MVP_MOMENT_BOUNDS

// Specialization of the primal and adjoint in Worchel & Alexa 2025, section 5.1.
// Status codes match the Unity CPU implementation.
struct MbdState
{
    float3 diagonal;
    float3 lower; // l10, l20, l21
    float3 polynomial;
    float3 nodes;
    float3 weights;
    float3 mask;
    float3 designated;
    int count;
    uint status;
    float bound;
};

float3 MbdSolve(float3 d, float3 l, float3 rhs)
{
    float3 x;
    float y0 = rhs.x / d.x;
    float y1 = (rhs.y - l.x * y0) / d.y;
    float y2 = (rhs.z - (l.y * y0 + l.z * y1)) / d.z;
    x.z = y2 / d.z;
    x.y = (y1 - l.z * x.z) / d.y;
    x.x = (y0 - (l.x * x.y + l.y * x.z)) / d.x;
    return x;
}

void MbdSortPair(inout float2 a, inout float2 b)
{
    if (abs(a.x) <= abs(b.x)) return;
    float2 old = a;
    a = b;
    b = old;
}

bool MbdValid(uint status) { return status <= 1; }

MbdState MbdPrimal(float m0, float4 moments, float eta, float alpha, float beta, int order)
{
    MbdState result = (MbdState)0;
    if (!isfinite(m0) || !all(isfinite(moments)) || !isfinite(eta))
    {
        result.status = 5;
        return result;
    }
    precise float mass = (1.0 - alpha) * m0 + alpha;
    precise float4 m = (1.0 - alpha) * moments + alpha * float4(0, order == 2 ? 0.375 : 1.0, 0, 0.375);
    result.status = 2;
    if (!(mass > 0)) return result;
    precise float d0 = sqrt(mass);
    precise float l10 = m.x / d0;
    precise float leg1 = m.y - l10 * l10;
    if (!(leg1 > 0)) return result;
    precise float d1 = sqrt(leg1);
    precise float l20 = 0;
    precise float l21 = 0;
    precise float d2 = 1;
    if (order == 2)
    {
        l20 = m.y / d0;
        l21 = (m.z - l20 * l10) / d1;
        precise float leg2 = m.w - (l20 * l20 + l21 * l21);
        if (!(leg2 > 0)) return result;
        d2 = sqrt(leg2);
    }
    result.diagonal = float3(d0, d1, d2);
    result.lower = float3(l10, l20, l21);
    float3 c = MbdSolve(result.diagonal, result.lower, float3(1, eta, order == 2 ? eta * eta : 0));
    result.polynomial = c;
    if (!all(isfinite(c))) { result.status = 5; return result; }

    result.status = 0;
    float2 a = float2(eta, 1);
    float2 b = 0;
    float2 f = 0;
    int count = order + 1;
    if (order == 1)
    {
        b.x = -c.x / c.y;
        if (!isfinite(b.x)) count = 1;
    }
    else
    {
        // Stable quadratic formula using a compensated difference of products.
        precise float a4 = 4.0 * c.z;
        precise float product = a4 * c.x;
        precise float error = mad(-a4, c.x, product);
        precise float disc = mad(c.y, c.y, -product) + error;
        if (disc < 0) { result.status = 3; return result; }
        float radical = sqrt(disc);
        float q = -0.5 * (c.y + (c.y < 0 ? -radical : radical));
        if (q == 0) { result.status = 4; return result; }
        b.x = c.x / q;
        f.x = q / c.z;
        if (!isfinite(b.x)) { result.status = 5; return result; }
        if (!isfinite(f.x)) count = 2;
    }
    if (count != order + 1) result.status = 1;
    if (count >= 2) MbdSortPair(a, b);
    if (count == 3)
    {
        MbdSortPair(b, f);
        MbdSortPair(a, b);
    }
    result.count = count;
    result.nodes = float3(a.x, b.x, f.x);
    result.designated = float3(a.y, b.y, f.y);
    result.mask = float3(a.y == 1 ? beta : a.x < eta, b.y == 1 ? beta : b.x < eta, f.y == 1 ? beta : f.x < eta);

    // The two- and three-node Björck–Pereyra steps, with the infinite node removed.
    float3 weights = float3(mass, 0, 0);
    if (count >= 2)
    {
        float ba = b.x - a.x;
        if (ba == 0) { result.status = 4; return result; }
        float m1 = mad(-a.x, mass, m.x);
        if (count == 3)
        {
            float ca = f.x - a.x;
            float cb = f.x - b.x;
            if (ca == 0 || cb == 0) { result.status = 4; return result; }
            float transformed = mad(-b.x, m1, mad(-a.x, m.x, m.y)) / ca;
            weights.z = transformed / cb;
            m1 -= transformed;
        }
        float intermediate = m1 / ba;
        weights.x = mass - intermediate;
        weights.y = intermediate - weights.z;
    }
    result.weights = weights;
    result.bound = dot(result.mask, weights);
    if (!isfinite(result.bound)) result.status = 5;
    return result;
}

void MbdRootReverse(float x, float w, float designated, float3 dual, float3 c, inout float3 dc, inout float de)
{
    float dp = dual.y + 2.0 * dual.z * x;
    if (designated != 0)
        de -= w * dp;
    else if (abs(w) > 5.5e-20)
    {
        float dk = c.y + 2.0 * c.z * x;
        float ratio = dp / dk;
        float wx = w * x;
        dc += ratio * float3(w, wx, wx * x);
    }
}

uint MbdAdjoint(MbdState s, float eta, float alpha, int order, out float4 powerGradient, out float2 massAndDepth)
{
    powerGradient = 0;
    massAndDepth = 0;
    if (!MbdValid(s.status)) return s.status;
    float3 p = float3(s.mask.x, 0, 0);
    if (s.count >= 2)
    {
        float slope = (s.mask.y - s.mask.x) / (s.nodes.y - s.nodes.x);
        if (s.count == 3)
        {
            float second = (s.mask.z - s.mask.y) / (s.nodes.z - s.nodes.y);
            p.z = (second - slope) / (s.nodes.z - s.nodes.x);
            slope = mad(-s.nodes.y, p.z, slope);
        }
        p.x = mad(-s.nodes.x, slope, p.x);
        p.y = mad(-s.nodes.x, p.z, slope);
    }
    float3 dc = 0;
    float de = 0;
    MbdRootReverse(s.nodes.x, s.weights.x, s.designated.x, p, s.polynomial, dc, de);
    if (s.count >= 2) MbdRootReverse(s.nodes.y, s.weights.y, s.designated.y, p, s.polynomial, dc, de);
    if (s.count == 3) MbdRootReverse(s.nodes.z, s.weights.z, s.designated.z, p, s.polynomial, dc, de);
    if (order == 1) dc.z = 0;
    float3 u = MbdSolve(s.diagonal, s.lower, dc);
    float3 c = s.polynomial;
    powerGradient = (1.0 - alpha) * float4(
        p.y - (u.x * c.y + u.y * c.x),
        p.z - (u.x * c.z + u.y * c.y + u.z * c.x),
        -(u.y * c.z + u.z * c.y), -u.z * c.z);
    massAndDepth = float2((1.0 - alpha) * (p.x - u.x * c.x), de + u.y + 2.0 * eta * u.z);
    return all(isfinite(powerGradient)) && all(isfinite(massAndDepth)) ? s.status : 5;
}

#endif
