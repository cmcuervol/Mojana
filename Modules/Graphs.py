import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as colors



# =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
#                                    cmaps
# =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*


def cmapeador(colrs=None, levels=None, name='coqueto'):
    """
    Make a new color map with the colors and levels given, adapted from Julian Sepulveda & Carlos Hoyos

    IMPUTS
    colrs  : list of tuples of RGB colors combinations
    levels : numpy array of levels correspond to each color in colors
    name   : name to register the new cmap

    OUTPUTS
    cmap   : color map
    norm   : normalization with the levesl given optimized for the cmap
    """
    if colrs == None:
        colrs = [(255, 255, 255),(0, 255, 255), (0, 0, 255),(70, 220, 45),(44, 141, 29),\
                  (255,255,75),(255,142,0),(255,0,0),(128,0,128),(102,0,102),\
                  (255, 153, 255)]

    if levels == None:
        levels = np.array([0.,1.,5.,10.,20.,30.,45.,60., 80., 100., 150.])
    # print levels
    scale_factor   = ((255-0.)/(levels.max() - levels.min()))
    new_Limits     = list(np.array(np.round((levels-levels.min()) * scale_factor/255.,3),dtype=float))
    Custom_Color   = map(lambda x: tuple(ti/255. for ti in x) , colrs)
    nueva_tupla    = [((new_Limits[i]),Custom_Color[i],) for i in range(len(Custom_Color))]
    cmap_new       = colors.LinearSegmentedColormap.from_list(name,nueva_tupla)
    levels_nuevos  = np.linspace(np.min(levels),np.max(levels),255)
    # print levels_nuevos
    # print new_Limits
    # levels_nuevos  = np.linspace(np.min(levels),np.max(levels),1000)
    norm_new       = colors.BoundaryNorm(boundaries=levels_nuevos, ncolors=256)
    # norm           = colors.BoundaryNorm(boundaries=levels_nuevos, ncolors=1000)

    return cmap_new, norm_new


def newjet(cmap="jet"):
    """
    function to make a newd colorbar with white at center
    IMPUTS
    cmap: colormap to change
    RETURNS
    newcmap : colormap with white as center
    """
    jetcmap = plt.cm.get_cmap(cmap, 11) #generate a jet map with 11 values
    jet_vals = jetcmap(np.arange(11)) #extract those values as an array
    jet_vals[5] = [1, 1, 1, 1] #change the middle value
    newcmap = colors.LinearSegmentedColormap.from_list("newjet", jet_vals)
    return newcmap


class MidpointNormalize(colors.Normalize):
    """
    New Normalization with a new parameter: midpoint
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


class SqueezedNorm(matplotlib.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, mid=0, s1=1, s2=1, clip=False):
        self.vmin = vmin # minimum value
        self.mid  = mid  # middle value
        self.vmax = vmax # maximum value
        self.s1=s1; self.s2=s2
        f = lambda x, zero,vmax,s: np.abs((x-zero)/(vmax-zero))**(1./s)*0.5
        self.g = lambda x, zero,vmin,vmax, s1,s2: f(x,zero,vmax,s1)*(x>=zero) - \
                                             f(x,zero,vmin,s2)*(x<zero)+0.5
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        r = self.g(value, self.mid,self.vmin,self.vmax, self.s1,self.s2)
        return np.ma.masked_array(r)


# fig, (ax, ax2, ax3) = plt.subplots(nrows=3,
#                                    gridspec_kw={"height_ratios":[3,2,1], "hspace":0.25})
#
# x = np.linspace(-13,4, 110)
# norm=SqueezedNorm(vmin=-13, vmax=4, mid=0, s1=1.7, s2=4)
#
# line, = ax.plot(x, norm(x))
# ax.margins(0)
# ax.set_ylim(0,1)
#
# im = ax2.imshow(np.atleast_2d(x).T, cmap="Spectral_r", norm=norm, aspect="auto")
# cbar = fig.colorbar(im ,cax=ax3,ax=ax2, orientation="horizontal")


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, min_val=None, max_val=None, name='shiftedcmap'):
    """
    Function to offset the "center" of a colormap. Useful for data with a \
    negative min and positive max and you want the middle of the colormap's dynamic \
    range to be at zero. Adapted from https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib

    IMPUTS
    -----
        cmap  : The matplotlib colormap to be altered.
        start : Offset from lowest point in the colormap's range.
                Defaults to 0.0 (no lower ofset). Should be between
                0.0 and `midpoint`.
        midpoint : The new center of the colormap. Defaults to
                   0.5 (no shift). Should be between 0.0 and 1.0. In
                   general, this should be  1 - vmax/(vmax + abs(vmin))
                   For example if your data range from -15.0 to +5.0 and
                   you want the center of the colormap at 0.0, `midpoint`
                   should be set to  1 - 5/(5 + 15)) or 0.75
        stop : Offset from highets point in the colormap's range.
               Defaults to 1.0 (no upper ofset). Should be between
               `midpoint` and 1.0.
        min_val : mimimun value of the dataset,
                  only use when 0.0 is pretend to be the midpoint of the colormap
        max_val : maximun value of the dataset,
                  only use when 0.0 is pretend to be the midpoint of the colormap
        name    : Name of the output cmap

    """
    epsilon = 0.001
    # start, stop = 0.0, 1.0
    if min_val is not None and max_val is not None:
        min_val, max_val = min(0.0, min_val), max(0.0, max_val)
        midpoint = 1.0 - max_val/(max_val + abs(min_val))

    cdict = {'red': [], 'green': [], 'blue': [], 'alpha': []}
    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)
    # shifted index to match the data
    shift_index = np.hstack([np.linspace(0.0, midpoint, 128, endpoint=False), \
                            np.linspace(midpoint, 1.0, 129, endpoint=True)])
    for ri, si in zip(reg_index, shift_index):
        if abs(si - midpoint) < epsilon:
            r, g, b, a = cmap(0.5) # 0.5 = original midpoint.
        else:
            r, g, b, a = cmap(ri)
        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))
    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)
    return newcmap
