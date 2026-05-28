import numpy as np
import matplotlib.pyplot as plt

def pretty_names(name):
    nice_names = {
        "noise": "Noise",
        "sine_gaussian": "Sine-Gaussian Glitch",
        "ringdown": "Ringdown Glitch",
    }
    return nice_names.get(name, name)

def plot_signal(
    t,
    signal,
    ax=None,
    title: str = "Signal",
    xlabel: str = "Time [s]",
    ylabel: str = "Amplitude",
    color: str = None,
    show: bool = True,
    ):
    """
    Plot a time-domain signal.

    Parameters:
    - t: 1D array of time values.
    - signal: 1D array of signal amplitudes.
    - ax: Optional matplotlib axis to draw on.
    - title: Plot title.
    - xlabel: Label for x-axis.
    - ylabel: Label for y-axis.
    - show: Whether to call plt.show().

    Returns:
    - fig, ax: Matplotlib figure and axis objects.
    """
    t = np.asarray(t)
    signal = np.asarray(signal)

    if t.ndim != 1 or signal.ndim != 1:
        raise ValueError("'t' and 'signal' must be 1D arrays.")
    if len(t) != len(signal):
        raise ValueError("'t' and 'signal' must have the same length.")
    
    created_figure = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
        created_figure = True
    else:
        fig = ax.figure

    ax.plot(t, signal, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
       
    if show and created_figure:
        plt.tight_layout()
        plt.show() 

    return fig, ax  


def plot_spectrogram(
    f,
    t_spec,
    Sxx_db,
    ax=None,
    title: str = "Spectrogram",
    xlabel: str = "Time [s]",
    ylabel: str = "Frequency [Hz]",
    colorbar: bool = True,
    show: bool = True,
):
    """
    Plot a spectrogram.

    Parameters:
    - f: 1D array of frequencies.
    - t_spec: 1D array of spectrogram time bins.
    - Sxx_db: 2D spectrogram array (freq x time).
    - ax: Optional matplotlib axis to draw on.
    - title: Plot title.
    - xlabel: Label for x-axis.
    - ylabel: Label for y-axis.
    - colorbar: Whether to display a colorbar.
    - show: Whether to call plt.show().

    Returns:
    - fig, ax: Matplotlib figure and axis objects.
    """
    f = np.asarray(f)
    t_spec = np.asarray(t_spec)
    Sxx_db = np.asarray(Sxx_db)

    if f.ndim != 1 or t_spec.ndim != 1:
        raise ValueError("'f' and 't_spec' must be 1D arrays.")
    if Sxx_db.ndim != 2:
        raise ValueError("'Sxx_db' must be a 2D array.")
    if Sxx_db.shape != (len(f), len(t_spec)):
        raise ValueError(
            "'Sxx_db' shape must match (len(f), len(t_spec))."
        )

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
        created_fig = True
    else:
        fig = ax.figure

    mesh = ax.pcolormesh(t_spec, f, Sxx_db, shading='gouraud') # shading='auto' can be used for discrete visualization, while shading='gouraud' uses a smooth shading between cells.
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if colorbar:
        fig.colorbar(mesh, ax=ax, label="Log-Magnitude [dB]")

    if show and created_fig:
        plt.tight_layout()
        plt.show()

    return fig, ax


def plot_sample(
    t,
    signal,
    f,
    t_spec,
    Sxx_db,
    label=None,
    metadata: dict = None,
    figsize=(10, 12),
    show: bool = True,
    show_params: bool = False,
):
    """
    Plot a signal and its spectrogram together.

    Parameters:
    - t: 1D array of time values.
    - signal: 1D array of signal amplitudes.
    - f: 1D array of frequencies.
    - t_spec: 1D array of spectrogram time bins.
    - Sxx_db: 2D spectrogram array.
    - label: Optional label for the sample.
    - metadata: Optional metadata dictionary.
    - figsize: Figure size.
    - show: Whether to call plt.show().
    - show_params: Whether to show t0 and f0 params with lines in the spectrogram.

    Returns:
    - fig, axes: Matplotlib figure and axes.
    """
    sample_title = "" #Dinamic title

    glitch_type = None
    glitch_metadata = None

    if metadata is not None:
        glitch_type = metadata.get("glitch_type", "Unknown type") #Here we use get in case the key is not in the dictionary, it will return 'unknown' (safer than metadata["glitch_type"])
        glitch_metadata = metadata.get("glitch_metadata", None)
        sample_title += f" {pretty_names(glitch_type)}  "

    #if label is not None:
    #    title_suffix += f" | label:{label}"

    fig, axes = plt.subplots(2, 1, figsize=figsize)

    plot_signal(
        t,
        signal,
        ax=axes[0],
        title=f"{sample_title}",
        show=False,
    )

    plot_spectrogram(
        f,
        t_spec,
        Sxx_db,
        ax=axes[1],
        title="Spectrogram",
        show=False,
    )

    # ------------------------------------------------ #
    # Overlay glitch metadata if available
    # ------------------------------------------------ #
    if glitch_metadata is not None and show_params: 
        t0 = glitch_metadata.get("t0", None)
        f0 = glitch_metadata.get("f0", None)
        sigma = glitch_metadata.get("sigma", None)
        tau = glitch_metadata.get("tau", None)

    # Vertical line at glitch time
        if t0 is not None:
            axes[0].axvline(t0, linestyle="--", color="firebrick", linewidth=1.5, label=f"$t_0$")
            axes[1].axvline(t0, linestyle="--", color="firebrick", linewidth=1.5, label=f"$t_0$")

        # Horizontal line at central frequency
        if f0 is not None:
            axes[1].axhline(f0, linestyle="-.", color="firebrick", linewidth=1.5, label=f"$f_0$")   

        param_lines = []


        if t0 is not None:
            param_lines.append(f"$t_0={t0:.2f}$ s")
        if f0 is not None:
            param_lines.append(f"$f_0={f0:.2f}$ Hz")
        if sigma is not None:
            param_lines.append(f"$\\sigma={sigma:.2f}$ s")
        if tau is not None:
            param_lines.append(f"$\\tau={tau:.2f}$ s")

        param_text = "\n".join(param_lines)

        if param_text:
            for ax in axes: 
                ax.text(
                    0.98, 0.95,
                    param_text,
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="palegoldenrod",
                        alpha=0.8,
                        edgecolor="black"
                    ),
                )

        axes[0].legend()
        axes[1].legend()
    
        # # Highlight temporal support for sine-gaussian
        # if glitch_type == "sine_gaussian" and sigma is not None and t0 is not None:
        #     axes[0].axvspan(t0 - 3 * sigma, t0 + 3 * sigma, alpha=0.2)
        #     axes[1].axvspan(t0 - 3 * sigma, t0 + 3 * sigma, alpha=0.2)

        # # Highlight temporal support for ringdown
        # if glitch_type == "ringdown" and tau is not None and t0 is not None:
        #     axes[0].axvspan(t0, t0 + 3 * tau, alpha=0.2)
        #     axes[1].axvspan(t0, t0 + 3 * tau, alpha=0.2)

        ## Including the parameters in the plot
        

    plt.tight_layout()

    if show:
        plt.show()

    return fig, axes

