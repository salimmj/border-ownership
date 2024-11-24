import marimo

__generated_with = "0.9.23"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def __():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def __():
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    from jax.scipy.signal import convolve2d

    return convolve2d, jax, jnp, plt


@app.cell(hide_code=True)
def __(jnp, plt):
    def drawnu(col):
        """
        Adds a U-shaped figure outline to a plot.
        """
        halfhole = 2.0  # Size of the center hole
        halfspan = 6.0  # Half-span of the figure

        left = -halfspan
        ileft = -halfhole
        iright = halfhole
        right = halfspan
        top = -halfspan
        bot = halfspan
        bail = halfhole

        cx = [left, left, ileft, ileft, iright, iright, right, right, left]
        cy = [bot, top, top, bail, bail, top, top, bot, bot]

        for j in range(len(cx)-1):
            plt.plot([cx[j], cx[j+1]], [cy[j], cy[j+1]], color=col)

    def drawcheck(col):
        """
        Adds checkerboard grid lines to a plot.
        """
        x = jnp.arange(-18, 19, 4)
        y = jnp.arange(-6, 11, 8)
        for xi in x:
            plt.axvline(x=xi, color=col)
        for yi in y:
            plt.axhline(y=yi, color=col)

    return drawcheck, drawnu


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # Figure 1: Visualization of the U-shaped Figure
        This figure displays the U-shaped stimulus used to model figure-ground segregation in the visual cortex. The U-shaped figure is created by defining a large square and subtracting an inner rectangle, resulting in a shape with a central hole. In the context of the paper, this stimulus represents a complex object whose borders need to be processed by the visual system to determine border ownership.
        """
    )
    return


@app.cell(hide_code=True)
def __(jnp, plt):
    # The 'ramps' variable selects the connection scheme.
    ramps = 0  # Set to 0 for post-hoc regression weights, 1 for a priori linear ramps.

    # Master parameters
    nPix = 200       # Number of pixels in the complete field
    wdeg = 25.0      # Size of the complete field in degrees
    fgsz = 4.0       # Size of the figure in degrees
    pixperdeg = nPix / wdeg
    halfhole = 2.0   # Size of the center hole for the U-shaped figure
    halfspan = 6.0   # Half-span for the U-shaped figure

    # Coordinate system and figure logical matrix
    XV = jnp.linspace(-wdeg/2, wdeg/2, nPix)
    # Corrected meshgrid call
    XV_plus1 = jnp.linspace(-wdeg/2, wdeg/2, nPix + 1)
    X_full, Y_full = jnp.meshgrid(XV_plus1, XV_plus1)
    X = X_full[:-1, :-1]
    Y = Y_full[:-1, :-1]
    nu = 1  # Set to 1 for U-shaped figure, 0 for square figure

    if nu:
        # U-shaped figure
        figcx = ((X > (-fgsz / (fgsz / halfhole))) & (X < (fgsz / (fgsz / halfhole)))) & \
                ((Y > (-fgsz / (fgsz / halfspan))) & (Y < (fgsz / (fgsz / halfhole))))
    else:
        # Simple square figure
        figcx = ((X > (-fgsz / (fgsz / halfhole))) & (X < (fgsz / (fgsz / halfhole)))) & \
                ((Y > (-fgsz / (fgsz / halfhole))) & (Y < (fgsz / (fgsz / halfhole))))

    # Create the figure by subtracting the inner part from the outer part
    figcx2 = ((X > (-fgsz / (fgsz / halfspan))) & (X < (fgsz / (fgsz / halfspan)))) & \
             ((Y > (-fgsz / (fgsz / halfspan))) & (Y < (fgsz / (fgsz / halfspan))))
    figcx = figcx2.astype(int) - figcx.astype(int)

    # Visualize the figure
    plt.figure()
    plt.imshow(figcx, extent=(XV[0], XV[-1], XV[0], XV[-1]), origin='lower')
    plt.title('Figure')
    plt.axis('equal')
    plt.colorbar()
    plt.gcf()
    return (
        X,
        XV,
        XV_plus1,
        X_full,
        Y,
        Y_full,
        fgsz,
        figcx,
        figcx2,
        halfhole,
        halfspan,
        nPix,
        nu,
        pixperdeg,
        ramps,
        wdeg,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # Figure 2: Visualization of the Checkerboard
        This figure shows the checkerboard pattern that serves as a control stimulus. The checkerboard consists of alternating black and white rectangles arranged in a grid. It contains similar local features (edges and orientations) as the U-shaped figure but does not form a coherent object. In the paper, the checkerboard is used to compare neural responses to figure versus non-figure stimuli, highlighting the role of border-ownership cells in figure-ground segregation.
        """
    )
    return


@app.cell(hide_code=True)
def __(X, XV, Y, figcx, jnp, plt):
    # Make a checkerboard
    check = jnp.ones_like(figcx)
    xo = jnp.arange(-16, 17, 8)
    y = jnp.arange(-10, 15, 8)
    for j in range(len(y)):
        if j == 0 or j == 2:
            x = xo + 4
        else:
            x = xo
        for i in range(len(x)):
            cond = ((X > x[i]-2) & (X <= x[i]+2)) & ((Y > y[j]-4) & (Y <= y[j]+4))
            check = jnp.where(cond, 0, check)

    # Visualize the checkerboard
    plt.figure()
    plt.imshow(check, extent=(XV[0], XV[-1], XV[0], XV[-1]), origin='lower')
    plt.title('Checkerboard')
    plt.axis('equal')
    plt.colorbar()
    plt.gcf()
    return check, cond, i, j, x, xo, y


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # Figure 3: V4 Responses to the U-shaped Figure
        This set of four subplots illustrates the responses of border-ownership (BO) tuned cells in area V4 to the U-shaped figure. Each subplot corresponds to BO cells preferring a different figure direction:

        - Preferring Figure DOWN
        - Preferring Figure RIGHT
        - Preferring Figure UP
        - Preferring Figure LEFT

        The color intensity represents the strength of the neural response after convolving the U-shaped figure with the receptive field templates of the BO cells. This figure demonstrates how BO-tuned V4 neurons encode the location of the figure relative to their receptive fields, supporting the paper's hypothesis about the role of V4 in figure-ground modulation.
        """
    )
    return


@app.cell(hide_code=True)
def __(XV, check, convolve2d, drawnu, figcx, jnp, pixperdeg, plt):
    # V4 cells: Define receptive fields for BO-tuned cells
    size_v4_x = int(4 * pixperdeg)
    size_v4_y = int(4 * pixperdeg)
    V4 = []

    # Create RF templates for each BO preference
    part1 = -1 * jnp.ones((int(2 * pixperdeg), int(4 * pixperdeg)))
    part2 = jnp.ones((int(2 * pixperdeg), int(4 * pixperdeg)))
    V4_1 = jnp.vstack((part1, part2))  # Prefers figure DOWN
    V4.append(V4_1)

    V4_2 = V4_1.T  # Prefers figure RIGHT
    V4.append(V4_2)

    V4_3 = jnp.flipud(V4_1)  # Prefers figure UP
    V4.append(V4_3)

    V4_4 = jnp.fliplr(V4_2)  # Prefers figure LEFT
    V4.append(V4_4)

    # Stack V4 templates into a 3D array
    V4 = jnp.stack(V4, axis=-1)

    # Convolve V4 templates with the figure and checkerboard
    M_list = []
    C_list = []
    for q in range(4):
        V4_rot = jnp.rot90(V4[:, :, q], 2)
        M_q = convolve2d(figcx.astype(float), V4_rot, mode='same')
        M_q = 2.0 * jnp.abs(M_q) + M_q
        M_list.append(M_q)

        buf = convolve2d(check.astype(float), V4_rot, mode='same')
        C_q = 2.0 * jnp.abs(buf)
        C_list.append(C_q)

    M = jnp.stack(M_list, axis=-1)
    C = jnp.stack(C_list, axis=-1)

    # Titles for plotting
    titl = ['Pref Figure DOWN', 'Pref Figure RIGHT', 'Pref Figure UP', 'Pref Figure LEFT']

    # Visualize V4 responses to the figure
    plt.figure()
    for _q in range(4):
        plt.subplot(2, 2, _q+1)
        plt.imshow(M[:, :, _q], extent=(XV[0], XV[-1], XV[0], XV[-1]), origin='lower')
        plt.axis('off')
        plt.title(titl[_q])
        drawnu([0, 0, 0])
        plt.axis('equal')
    plt.gcf()
    return (
        C,
        C_list,
        C_q,
        M,
        M_list,
        M_q,
        V4,
        V4_1,
        V4_2,
        V4_3,
        V4_4,
        V4_rot,
        buf,
        part1,
        part2,
        q,
        size_v4_x,
        size_v4_y,
        titl,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # Figure 4: V4 Responses to the Checkerboard
        Similar to Figure 3, this set of subplots shows the responses of BO-tuned V4 cells to the checkerboard stimulus. The responses are generally weaker and less localized compared to those elicited by the U-shaped figure. This illustrates that BO cells are more responsive to coherent figures than to non-figure patterns, aligning with the findings in the paper regarding the selective activation of BO-tuned cells by figures.
        """
    )
    return


@app.cell(hide_code=True)
def __(C, XV, drawcheck, plt, titl):
    # Visualize V4 responses to the checkerboard
    plt.figure()
    for _q in range(4):
        plt.subplot(2, 2, _q+1)
        plt.imshow(C[:, :, _q], extent=(XV[0], XV[-1], XV[0], XV[-1]), origin='lower')
        plt.axis('off')
        plt.title('Check ' + titl[_q])
        drawcheck([0, 0, 0])
        plt.axis('equal')
    plt.gcf()
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # Figure 5: Visualization of the Connection Scheme Between V4 and V1
        This figure displays the connection weights between V4 and V1 for each class of BO-tuned cells. The weights are derived from regression coefficients based on noise correlation data, representing the strength of feedback connections from V4 to V1. Each subplot corresponds to one BO preference direction:

        1. Preferring Figure DOWN
        2. Preferring Figure RIGHT
        3. Preferring Figure UP
        4. Preferring Figure LEFT

        In the context of the paper, this connection scheme embodies the proposed model where BO-tuned cells in V4 send positive feedback to V1 neurons in the direction of their preferred figure location and negative feedback in the opposite direction.
        """
    )
    return


@app.cell(hide_code=True)
def __(jnp, pixperdeg, plt, ramps, titl):
    # Define connection scheme between V4 and V1
    v4sz = 4
    xv = jnp.linspace(-8, 8, int(v4sz * 2 * pixperdeg) + 1)
    x_full, y_full = jnp.meshgrid(xv, xv)
    _x = x_full[:-1, :-1]
    _y = y_full[:-1, :-1]
    xv = xv[:-1]

    if ramps == 1:
        # A priori linear ramps (not implemented here)
        pass
    else:
        # Post-hoc regression weights
        z = jnp.hypot(_x, _y)  # Eccentricity
        a = jnp.abs(jnp.arctan2(_y, _x))  # Absolute angle
        B = jnp.array([-0.0508, -0.0404, 0.0119, 0.2387])  # Regression coefficients
        _buf = B[0]*a + B[1]*z + B[3]
        _buf = jnp.where(z > 4, 0, _buf)  # Zero connections beyond 4 degrees

        # Rotate _buf to create connection weights for each BO preference
        w1 = jnp.rot90(_buf, 3)  # Prefers figure DOWN
        w2 = _buf                # Prefers figure RIGHT
        w3 = jnp.rot90(_buf, 1)  # Prefers figure UP
        w4 = jnp.rot90(_buf, 2)  # Prefers figure LEFT

        w = jnp.stack([w1, w2, w3, w4], axis=-1)

    # Visualize the connection scheme
    plt.figure()
    for _q in range(4):
        plt.subplot(2, 2, _q+1)
        plt.imshow(w[:, :, _q], extent=(xv[0], xv[-1], xv[0], xv[-1]), origin='lower')
        plt.axis('off')
        plt.title(titl[_q])
        plt.colorbar()
        plt.axis('equal')
    plt.gcf()
    return B, a, v4sz, w, w1, w2, w3, w4, x_full, xv, y_full, z


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # Figure 6: Final V1 Responses After Feedback to the U-shaped Figure
        These subplots depict the V1 neural responses to the U-shaped figure after incorporating feedback from V4. The responses are modulated by the connection weights and show enhanced activity on the figure side and suppressed activity on the background side. This mirrors the figure-background modulation (FBM) observed in V1 neurons, as discussed in the paper, supporting the idea that feedback from BO-tuned V4 cells contributes to FBM in V1.
        """
    )
    return


@app.cell(hide_code=True)
def __(C, M, XV, convolve2d, drawnu, jnp, plt, titl, w):
    # Calculate the connection strength for each class of BO cells
    Mcs_list = []
    Ccs_list = []
    for _q in range(4):
        Mcs_q = convolve2d(M[:, :, _q], w[:, :, _q], mode='same')
        Mcs_list.append(Mcs_q)
        Ccs_q = convolve2d(C[:, :, _q], w[:, :, _q], mode='same')
        Ccs_list.append(Ccs_q)

    Mcs = jnp.stack(Mcs_list, axis=-1)
    Ccs = jnp.stack(Ccs_list, axis=-1)

    # Visualize the final V1 responses after feedback
    plt.figure()
    for _q in range(4):
        plt.subplot(2, 2, _q+1)
        plt.imshow(Mcs[:, :, _q], extent=(XV[0], XV[-1], XV[0], XV[-1]), origin='lower')
        plt.axis('off')
        plt.title('FB ' + titl[_q])
        plt.colorbar()
        drawnu([0, 0, 0])
        plt.axis('equal')
    plt.gcf()
    return Ccs, Ccs_list, Ccs_q, Mcs, Mcs_list, Mcs_q


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # Figure 7: Final V1 Responses After Feedback to the Checkerboard
        Similar to Figure 6, these subplots show the V1 responses to the checkerboard after feedback from V4. The responses are more uniform and lack the figure-ground modulation seen with the U-shaped figure. This emphasizes that the feedback connectivity from BO-tuned V4 cells selectively enhances figure representation in V1, consistent with the experimental observations reported in the paper.
        """
    )
    return


@app.cell(hide_code=True)
def __(Ccs, XV, drawcheck, plt, titl):

    plt.figure()
    for _q in range(4):
        plt.subplot(2, 2, _q+1)
        plt.imshow(Ccs[:, :, _q], extent=(XV[0], XV[-1], XV[0], XV[-1]), origin='lower')
        plt.axis('off')
        plt.title('FB Chk ' + titl[_q])
        plt.colorbar()
        drawcheck([0, 0, 0])
        plt.axis('equal')
    plt.gcf()
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # Figure 8: Averaged V1 Responses to the U-shaped Figure
        This figure presents the averaged V1 response across all BO directions for the U-shaped figure. The enhanced responses highlight the role of V4 feedback in modulating V1 activity to emphasize figure regions. This aligns with the paper's findings on how border-ownership tuning in V4 influences figure-background perception in V1.
        """
    )
    return


@app.cell
def __(Ccs, Mcs, XV, drawnu, jnp, plt):
    # Average across BO directions
    output = jnp.mean(Mcs, axis=-1)
    output_check = jnp.mean(Ccs, axis=-1)

    # Visualize the averaged outputs
    plt.figure()
    plt.imshow(output, extent=(XV[0], XV[-1], XV[0], XV[-1]), origin='lower')
    plt.axis('off')
    plt.colorbar()
    drawnu([0, 0, 0])
    plt.title('U-shaped Figure')
    plt.axis('equal')
    plt.gcf()
    return output, output_check


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # Figure 9: Averaged V1 Responses to the Checkerboard
        This figure shows the averaged V1 response to the checkerboard, serving as a comparison to Figure 8. The uniform response indicates the absence of figure-ground modulation when a coherent figure is not present. This supports the paper's conclusion that BO-tuned feedback from V4 is critical for enhancing figure representation in V1.
        """
    )
    return


@app.cell(hide_code=True)
def __(XV, drawcheck, output_check, plt):
    plt.figure()
    plt.imshow(output_check, extent=(XV[0], XV[-1], XV[0], XV[-1]), origin='lower')
    plt.axis('off')
    plt.colorbar()
    drawcheck([0, 0, 0])
    plt.title('Checkerboard')
    plt.axis('equal')
    plt.gcf()
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # Figure 10: Cross-Section Comparison Between Figure and Checkerboard Responses
        This plot compares the V1 responses along a horizontal cross-section for both the U-shaped figure and the checkerboard. It shows:

        Blue Line: V1 response to the U-shaped figure.
        Green Line: V1 response to the checkerboard.
        Red Line: Difference between the two responses (modulation).
        The plot demonstrates enhanced responses at the edges of the figure and suppressed responses in background regions. This mirrors the patterns observed in physiological recordings, as reported in the paper, and underscores the role of BO-tuned feedback in generating figure-background modulation in V1.
        """
    )
    return


@app.cell(hide_code=True)
def __(XV, output, output_check, plt):
    # Checkerboard comparison through cross-section
    ix = (XV > -8) & (XV < 8)
    plt.figure()
    plt.plot(XV[ix], output[80, ix], 'b', label='Figure')
    plt.plot(XV[ix], output_check[80, ix], 'g', label='Checkerboard')
    plt.plot(XV[ix], output[80, ix] - output_check[80, ix], 'r', label='Modulation')
    plt.xlim([-12, 12])
    plt.legend()
    plt.title('Cross-Section Comparison')
    plt.xlabel('Position (degrees)')
    plt.ylabel('Response')
    plt.axvline(x=-6, color='k', linestyle='--')
    plt.axvline(x=-2, color='k', linestyle='--')
    plt.axvline(x=2, color='k', linestyle='--')
    plt.axvline(x=6, color='k', linestyle='--')
    plt.gcf()
    return (ix,)


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
