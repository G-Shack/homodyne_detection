import streamlit as st
import math
from scipy.special import genlaguerre, factorial
import gmpy2
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
from scipy.interpolate import interp1d, interp2d
from scipy.signal import convolve
from skimage import transform
import pandas as pd
import base64
from functools import partial

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Wigner Distribution Visualizer",
    page_icon="🔭",
    layout="wide",
)

# ─────────────────────────────────────────────
# UTILITY / PHYSICS FUNCTIONS  (all fixed)
# ─────────────────────────────────────────────

def norm_state_vec(v):
    """Normalize a state vector."""
    n = np.linalg.norm(v)
    return v / n if n != 0 else v


def coh(n, alpha):
    """Coherent state in Fock basis (uses gmpy2 for large factorials)."""
    base = np.exp(-np.abs(alpha) ** 2 / 2)
    arr = np.zeros(n, dtype=object)
    for i in range(n):
        arr[i] = alpha ** i / float(gmpy2.sqrt(gmpy2.fac(i)))
    return np.array(arr, dtype="complex128") * base


def rho_input(inp_, kind="state_vec"):
    """
    Build a density matrix from different state specifications.
    kind: 'state_vec' | 'coherent' | 'cat_states'
    FIX: removed unfinished 'mixed_state' branch; added proper normalisation.
    """
    if kind == "state_vec":
        if len(inp_) < 2:
            inp_ = np.array([1.0, 0.0])
        inp_ = norm_state_vec(np.array(inp_, dtype="complex128"))
        return np.outer(inp_, np.conj(inp_))

    elif kind == "coherent":
        N, alpha = inp_
        coh_state = coh(int(N), alpha)
        return np.outer(coh_state, np.conj(coh_state))

    elif kind == "cat_states":
        N, alphas, c_is = inp_
        norm_c_is = norm_state_vec(np.array(c_is, dtype="complex128"))
        cat_st = np.zeros(int(N), dtype="complex128")
        for c_i, alpha in zip(norm_c_is, alphas):
            cat_st += c_i * coh(int(N), alpha)
        cat_st = norm_state_vec(cat_st)
        return np.outer(cat_st, np.conj(cat_st))

    else:
        raise ValueError(f"Unknown kind '{kind}'. Choose: state_vec, coherent, cat_states")


def wigner_laguerre(rho, x_min=-10, x_max=10, p_min=-10, p_max=10, res=200, return_axes=False):
    """
    Wigner distribution from density matrix via generalised Laguerre polynomials.
    Refs:
      - Ulf Leonhardt, Measuring the Quantum States of Light, ch.5
      - arXiv:1909.02395v1 Appendix A
    FIX: guard against non-square rho; avoid division-by-zero in normalisation.
    """
    rho = np.array(rho)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("rho must be a square 2-D array.")

    x_vec = np.linspace(x_min, x_max, res)
    p_vec = np.linspace(p_min, p_max, res)
    X, P = np.meshgrid(x_vec, p_vec)
    A = X + 1j * P
    B = np.abs(A) ** 2
    W = np.zeros((res, res))

    for n in range(rho.shape[0]):
        if np.abs(rho[n, n]) > 1e-15:
            W += np.real(rho[n, n] * (-1) ** n * genlaguerre(n, 0)(2 * B))
        for m in range(0, n):
            if np.abs(rho[m, n]) > 1e-15:
                W += 2 * np.real(
                    rho[m, n]
                    * (-1) ** m
                    * np.sqrt(2 ** (n - m) * factorial(m) / factorial(n))
                    * genlaguerre(m, n - m)(2 * B)
                    * A ** (n - m)
                )

    W = W * np.exp(-B) / np.pi
    total = np.sum(np.abs(W))
    W = W / total if total != 0 else W

    if return_axes:
        return W, x_vec, p_vec
    return W


def wig_coherent(alpha, xmin, xmax, pmin, pmax, res=200, g=np.sqrt(2), return_vecs=False):
    """
    Analytic Gaussian Wigner distribution for a coherent state.
    FIX: removed st.write side-effect; silently clamps axis range instead.
    """
    span = max(4 * np.abs(alpha), 6.0)
    if (xmax - xmin) < span or (pmax - pmin) < span:
        half = span / 2
        xmin, xmax, pmin, pmax = -half, half, -half, half

    xvec = np.linspace(xmin, xmax, res)
    pvec = np.linspace(pmin, pmax, res)
    X, P = np.meshgrid(xvec, pvec)
    wig = np.exp(-(X - g * np.real(alpha)) ** 2 - (P + g * np.imag(alpha)) ** 2)
    norm_wig = wig / np.sum(wig)
    if return_vecs:
        return norm_wig, xvec, pvec
    return norm_wig


def wig_vac_squeezed(r, theta, res=200, return_axes=False):
    """Wigner distribution for a vacuum squeezed state."""
    xv = np.linspace(-10, 10, res)
    X, P = np.meshgrid(xv, xv)
    th = np.deg2rad(theta)
    wig = (
        np.exp(
            -2
            * (
                (X * np.cos(th) + P * np.sin(th)) ** 2 * np.exp(-2 * r)
                + (-X * np.sin(th) + P * np.cos(th)) ** 2 * np.exp(2 * r)
            )
        )
        * 2
        / np.pi
    )
    total = np.sum(np.abs(wig))
    wig = wig / total if total != 0 else wig
    if return_axes:
        return wig, xv, xv
    return wig


def wig_after_loss(arr, eta):
    """
    Scale (stretch) a Wigner distribution to simulate loss η < 1.
    FIX: clamp eta to (0,1]; handle edge case eta==1 gracefully.
    """
    if eta <= 0 or eta > 1:
        raise ValueError("eta must be in (0, 1].")
    if eta == 1.0:
        return arr.copy()

    n = arr.shape[0]
    a = int((n - 1) / eta)
    b = (a - n) // 2
    new_x = np.linspace(-2 * b, a, n)

    x = np.arange(n)
    interp = interp2d(x, x, arr, kind="cubic")
    return interp(new_x, new_x)


def wig_loss(wig_dis, eta, xvec, pvec):
    """
    Apply loss to a Wigner distribution via convolution with thermal noise kernel.
    FIX: separate the two loss steps clearly; guard against eta==1 (no loss).
    """
    if eta >= 1.0:
        return wig_dis.copy()
    scaled = wig_after_loss(wig_dis, np.sqrt(eta))
    X, P = np.meshgrid(xvec, pvec)
    s = eta / (1 - eta)
    kernel = np.exp(-s * (X ** 2 + P ** 2))
    kernel /= np.sum(kernel)
    return convolve(scaled, kernel, mode="same")


def radon_filter(size):
    """
    Ramp filter for filtered back-projection (taken from skimage.transform.iradon source).
    FIX: renamed from 'filter' to avoid shadowing Python built-in.
    """
    n = np.concatenate(
        (np.arange(1, size / 2 + 1, 2, dtype=int), np.arange(size / 2 - 1, 0, -2, dtype=int))
    )
    f = np.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2
    f_ = np.fft.ifft(f)
    return np.real(f_[:, np.newaxis] / np.max(f_))


def perform_interpolation(arr1, arr2, m=100, x_min=-5, x_max=5, padding=1.0, kind="cubic"):
    non_zero_indices = np.nonzero(arr1)
    filtered_arr1 = arr1[non_zero_indices]
    filtered_arr2 = arr2[non_zero_indices]
    x_new = np.linspace(padding * x_min, padding * x_max, m)
    interp_func = interp1d(filtered_arr2, filtered_arr1, kind=kind, bounds_error=False, fill_value=0)
    return x_new, interp_func(x_new)


def meas_data_2_hist(sim_data, theta, data_points, dat_min, dat_max, bins, m=360):
    full = np.zeros((m, theta))
    for i in range(theta):
        a, b = np.histogram(sim_data[i * data_points: (i + 1) * data_points], bins=bins)
        _, c = perform_interpolation(a, b, m=m, x_min=dat_min, x_max=dat_max)
        full[:, i] = np.abs(c)[::-1]
    return full


def irad(hist_2d, thetas):
    """Inverse Radon (filtered back-projection). FIX: uses renamed radon_filter."""
    all_thetas = np.linspace(0, 360, thetas)
    filt = radon_filter(hist_2d.shape[0])
    filtered_img = np.real(np.fft.ifft(np.fft.fft(hist_2d, axis=0) * filt, axis=0))
    final_img = np.zeros((hist_2d.shape[0], hist_2d.shape[0]))
    x_arr = np.arange(hist_2d.shape[0]) - hist_2d.shape[0] // 2
    x, p = np.mgrid[: hist_2d.shape[0], : hist_2d.shape[0]] - hist_2d.shape[0] // 2
    for col, theta in enumerate(all_thetas):
        final_img += partial(
            np.interp, xp=x_arr, fp=filtered_img[:, col], left=0, right=0
        )(-x * np.sin(np.deg2rad(theta)) - p * np.cos(np.deg2rad(theta)))
    return final_img


def sim_homodyne_data(
    wg, xv, theta_steps=180, ADC_bits=8, pts=100,
    need_elec_noise=True, elec_var=0.3, data_res=10
):
    """Simulate homodyne detector data from a Wigner distribution."""
    thetas = np.linspace(0, 359, theta_steps)
    mask = np.array([1 if x % data_res == 0 else 0 for x in range(2 ** ADC_bits * data_res)])
    all_data = np.zeros(thetas.shape[0] * pts)
    grid = np.linspace(xv[0], xv[-1], 2 ** ADC_bits * data_res)

    for i, t in enumerate(thetas):
        marginal = transform.rotate(wg, t).sum(0)
        f = interp1d(xv, marginal, bounds_error=False, fill_value=0)
        discrete_p = f(grid) * mask
        total = np.sum(np.abs(discrete_p))
        if total == 0:
            continue
        discrete_p = np.abs(discrete_p) / total
        all_data[i * pts: (i + 1) * pts] = np.random.choice(grid, p=discrete_p, size=pts)

    if need_elec_noise:
        elec_f = np.exp(-(xv) ** 2 / elec_var)
        elec_fun = interp1d(xv, elec_f, bounds_error=False, fill_value=0)
        elec_p = np.abs(elec_fun(grid) * mask)
        elec_p /= np.sum(elec_p)
        elec_noise = np.random.choice(grid, p=elec_p, size=thetas.shape[0] * pts)
        return all_data + elec_noise
    return all_data


def download_csv(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="sim_data.csv">📥 Download simulated data (CSV)</a>'


def plot_wigner(W, xvec, pvec, title="Wigner Distribution"):
    """Return a Plotly heatmap figure for a Wigner distribution."""
    fig = go.Figure(
        go.Heatmap(
            z=W,
            x=xvec,
            y=pvec,
            colorscale="RdBu_r",
            zmid=0,
            colorbar=dict(title="W(x,p)"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="x (quadrature)",
        yaxis_title="p (quadrature)",
        height=500,
    )
    return fig


# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────

st.title("🔭 Wigner Distribution Visualizer")
st.markdown(
    "Visualise quantum-optical phase-space distributions and simulate "
    "homodyne tomography data."
)

# ── Sidebar: state selection ──────────────────
st.sidebar.header("⚙️ State Settings")

state_type = st.sidebar.selectbox(
    "Quantum State",
    ["Coherent", "Vacuum Squeezed", "Fock / Custom (density matrix)", "Cat State"],
)

res = st.sidebar.slider("Grid resolution", 50, 300, 150, step=10)

# ── Sidebar: loss channel ─────────────────────
st.sidebar.header("📉 Loss Channel")
apply_loss = st.sidebar.checkbox("Apply loss", value=False)
eta = st.sidebar.slider("Transmissivity η", 0.01, 1.0, 0.9, step=0.01, disabled=not apply_loss)

# ── Main: state-specific controls ────────────
tabs = st.tabs(["Phase-Space Plot", "Homodyne Simulation"])

# ────────────────────────────────────────────
# COMPUTE WIGNER
# ────────────────────────────────────────────
W, xvec, pvec = None, None, None

if state_type == "Coherent":
    col1, col2 = st.columns(2)
    alpha_re = col1.number_input("Re(α)", value=2.0, step=0.1)
    alpha_im = col2.number_input("Im(α)", value=0.0, step=0.1)
    alpha = alpha_re + 1j * alpha_im

    xmin = st.sidebar.number_input("x min", value=-8.0)
    xmax = st.sidebar.number_input("x max", value=8.0)
    pmin = st.sidebar.number_input("p min", value=-8.0)
    pmax = st.sidebar.number_input("p max", value=8.0)

    W, xvec, pvec = wig_coherent(alpha, xmin, xmax, pmin, pmax, res=res, return_vecs=True)
    title = f"Coherent State  α = {alpha_re:.2f}+{alpha_im:.2f}i"

elif state_type == "Vacuum Squeezed":
    col1, col2 = st.columns(2)
    r = col1.slider("Squeezing parameter r", 0.0, 3.0, 1.0, step=0.05)
    theta_sq = col2.slider("Squeezing angle θ (°)", 0, 360, 0)

    W, xvec, pvec = wig_vac_squeezed(r, theta_sq, res=res, return_axes=True)
    title = f"Vacuum Squeezed  r={r:.2f}, θ={theta_sq}°"

elif state_type == "Fock / Custom (density matrix)":
    st.info(
        "Enter the state vector coefficients (real part only for simplicity). "
        "The vector will be normalised automatically."
    )
    n_fock = st.number_input("Fock space truncation N", min_value=2, max_value=50, value=10)
    raw = st.text_input(
        "State vector  (comma-separated, e.g. 0,0,1 for |2⟩)",
        value="0,1,0,0,0",
    )
    try:
        sv = np.array([float(x.strip()) for x in raw.split(",")])
        if len(sv) < n_fock:
            sv = np.pad(sv, (0, int(n_fock) - len(sv)))
        elif len(sv) > n_fock:
            sv = sv[: int(n_fock)]
        rho = rho_input(sv, kind="state_vec")
        W, xvec, pvec = wigner_laguerre(rho, res=res, return_axes=True)
        title = "Custom Fock-space State"
    except Exception as e:
        st.error(f"Error computing Wigner: {e}")

elif state_type == "Cat State":
    st.markdown("**Superposition of coherent states: |cat⟩ ∝ Σ cᵢ |αᵢ⟩**")
    n_fock = st.number_input("Fock space truncation N", min_value=5, max_value=80, value=30)
    n_cats = st.number_input("Number of component states", min_value=2, max_value=6, value=2)

    alphas, c_is = [], []
    cols = st.columns(int(n_cats))
    for k, col in enumerate(cols):
        with col:
            st.markdown(f"**|α_{k}⟩**")
            ar = st.number_input(f"Re(α_{k})", value=float(2 * ((-1) ** k)), key=f"ar{k}")
            ai = st.number_input(f"Im(α_{k})", value=0.0, key=f"ai{k}")
            ci = st.number_input(f"c_{k}", value=1.0, key=f"ci{k}")
            alphas.append(ar + 1j * ai)
            c_is.append(ci)

    try:
        rho = rho_input((int(n_fock), alphas, np.array(c_is, dtype="complex128")), kind="cat_states")
        W, xvec, pvec = wigner_laguerre(rho, res=res, return_axes=True)
        title = f"{int(n_cats)}-component Cat State"
    except Exception as e:
        st.error(f"Error computing Wigner: {e}")

# ────────────────────────────────────────────
# APPLY LOSS
# ────────────────────────────────────────────
if W is not None and apply_loss and eta < 1.0:
    with st.spinner("Applying loss channel…"):
        try:
            W = wig_loss(W, eta, xvec, pvec)
            title += f"  (η={eta:.2f})"
        except Exception as e:
            st.warning(f"Loss channel failed: {e}")

# ────────────────────────────────────────────
# TAB 1 — PHASE SPACE PLOT
# ────────────────────────────────────────────
with tabs[0]:
    if W is not None and xvec is not None:
        fig = plot_wigner(W, xvec, pvec, title=title)
        st.plotly_chart(fig, use_container_width=True)

        # Negativity indicator
        neg_vol = float(np.sum(W[W < 0]))
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Min W(x,p)", f"{W.min():.4f}")
        col_b.metric("Max W(x,p)", f"{W.max():.4f}")
        col_c.metric("Negativity volume", f"{neg_vol:.4f}")
        if neg_vol < -1e-6:
            st.success("✅ Wigner function has negative regions — non-classical state!")
        else:
            st.info("ℹ️ Wigner function is non-negative — classical-like state.")
    else:
        st.warning("Set parameters above to generate the Wigner distribution.")

# ────────────────────────────────────────────
# TAB 2 — HOMODYNE SIMULATION
# ────────────────────────────────────────────
with tabs[1]:
    st.subheader("Homodyne Tomography Simulation")

    if W is None or xvec is None:
        st.warning("Generate a Wigner distribution first (set parameters above).")
    else:
        col1, col2, col3 = st.columns(3)
        theta_steps = col1.number_input("Quadrature angles", min_value=4, max_value=360, value=36)
        pts = col2.number_input("Measurements per angle", min_value=10, max_value=1000, value=100)
        ADC_bits = col3.selectbox("ADC bits", [6, 8, 10, 12], index=1)

        st.markdown("**Electronic Noise**")
        col4, col5 = st.columns(2)
        need_elec = col4.checkbox("Add electronic noise", value=True)
        elec_var = col5.slider("Electronic noise variance", 0.05, 2.0, 0.3, disabled=not need_elec)

        if st.button("▶ Run Simulation"):
            with st.spinner("Simulating homodyne data…"):
                try:
                    sim_data = sim_homodyne_data(
                        W, xvec,
                        theta_steps=int(theta_steps),
                        ADC_bits=int(ADC_bits),
                        pts=int(pts),
                        need_elec_noise=need_elec,
                        elec_var=elec_var,
                    )

                    # ── Sinogram ──────────────────────────────────
                    st.subheader("Sinogram (quadrature histograms)")
                    bins = 64
                    sino = meas_data_2_hist(
                        sim_data,
                        theta=int(theta_steps),
                        data_points=int(pts),
                        dat_min=xvec[0],
                        dat_max=xvec[-1],
                        bins=bins,
                        m=bins,
                    )
                    fig_sino = px.imshow(
                        sino,
                        aspect="auto",
                        color_continuous_scale="Viridis",
                        labels={"x": "Quadrature angle index", "y": "Quadrature value"},
                        title="Sinogram",
                    )
                    st.plotly_chart(fig_sino, use_container_width=True)

                    # ── Reconstructed Wigner via inverse Radon ────
                    if int(theta_steps) >= 4:
                        st.subheader("Reconstructed Wigner (filtered back-projection)")
                        with st.spinner("Running inverse Radon…"):
                            recon = irad(sino, int(theta_steps))
                            fig_recon = plot_wigner(
                                recon,
                                np.linspace(xvec[0], xvec[-1], recon.shape[1]),
                                np.linspace(xvec[0], xvec[-1], recon.shape[0]),
                                title="Reconstructed Wigner",
                            )
                            st.plotly_chart(fig_recon, use_container_width=True)

                    # ── Download ──────────────────────────────────
                    thetas_arr = np.repeat(np.linspace(0, 359, int(theta_steps)), int(pts))
                    df = pd.DataFrame({"theta_deg": thetas_arr, "quadrature": sim_data})
                    st.markdown(download_csv(df), unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Simulation error: {e}")