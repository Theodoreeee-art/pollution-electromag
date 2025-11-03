# -*- coding: utf-8 -*-


# Python packages
import matplotlib.pyplot
import numpy
import os


# MRG packages
import _env


def myimshow(tab, **kwargs):
    """Customized plot."""

    if 'dpi' in kwargs and kwargs['dpi']:
        dpi = kwargs['dpi']
    else:
        dpi = 100

    # -- create figure
    fig = matplotlib.pyplot.figure(dpi=dpi)
    ax = matplotlib.pyplot.axes()

    if 'title' in kwargs and kwargs['title']:
        title = kwargs['title']
    if 'cmap' in kwargs and kwargs['cmap']:
        cmap = kwargs['cmap']
    else:
        cmap = 'jet'
    #if 'clim' in kwargs and kwargs['clim']:
    #    clim = kwargs['clim']
    if 'vmin' in kwargs and kwargs['vmin']:
        vmin = kwargs['vmin']
    if 'vmax' in kwargs and kwargs['vmax']:
        vmax = kwargs['vmax']

    # -- plot curves
    if 'cmap' in kwargs and kwargs['cmap']:
        matplotlib.pyplot.imshow(tab, cmap=cmap)
    else:
        matplotlib.pyplot.imshow(tab, cmap=cmap)
    if 'title' in kwargs and kwargs['title']:
        matplotlib.pyplot.title(title)
    else:
        matplotlib.pyplot.imshow(tab, cmap=cmap)
    if 'colorbar' in kwargs and kwargs['colorbar']:
        matplotlib.pyplot.colorbar()

#    if 'clim' in kwargs and kwargs['clim']:
#        matplotlib.pyplot.clim(clim)
    if 'vmin' in kwargs and kwargs['vmin']:
        matplotlib.pyplot.clim(vmin, vmax)

    if 'filename' in kwargs and kwargs['filename']:
        output_file = kwargs['filename']
        (root, ext) = os.path.splitext(output_file)
        matplotlib.pyplot.savefig(root + '_plot' + ext, format=ext[1:])
        matplotlib.pyplot.close()
    else:
        matplotlib.pyplot.show()
        matplotlib.pyplot.close()

    matplotlib.pyplot.close(fig)

    return


def _plot_uncontroled_solution(u, chi):
#def _plot_uncontroled_solution(x_plot, y_plot, x, y, u, chi):
    max_abs_re = float(max(abs(numpy.min(numpy.real(u))), abs(numpy.max(numpy.real(u)))))
    myimshow(numpy.real(u), title='$\operatorname{Re}(u_{0})$ in $\Omega$', colorbar='colorbar', cmap='jet', vmin=-max_abs_re, vmax=max_abs_re, filename='fig_u0_re.jpg')
    max_abs_im = float(max(abs(numpy.min(numpy.imag(u))), abs(numpy.max(numpy.imag(u)))))
    myimshow(numpy.imag(u), title='$\operatorname{Im}(u_{0})$ in $\Omega$', colorbar='colorbar', cmap='jet', vmin=-max_abs_im, vmax=max_abs_im, filename='fig_u0_im.jpg')
    chi_vmin = float(numpy.min(chi))
    chi_vmax = float(numpy.max(chi))
    myimshow(chi, title='$\chi_{0}$ in $\Omega$', colorbar='colorbar', cmap='jet', vmin=chi_vmin, vmax=chi_vmax, filename='fig_chi0_re.jpg')
    # k_begin = 0
    # k_end = len(x) - 1
    # for k in range(k_begin, k_end):
    #     x_plot[k] = k
    #     y_plot[k] = chi[int(y[k]), int(x[k])]
    # matplotlib.pyplot.plot(x_plot, y_plot)
    # matplotlib.pyplot.title('$\chi_{0}$ in $\Omega$')
    # matplotlib.pyplot.show()

    return


def _plot_controled_solution(u, chi):
#def _plot_controled_solution(x_plot, y_plot, x, y, u, chi):

    max_abs_re = float(max(abs(numpy.min(numpy.real(u))), abs(numpy.max(numpy.real(u)))))
    myimshow(numpy.real(u), title='$\operatorname{Re}(u_{n})$ in $\Omega$', colorbar='colorbar', cmap='jet', vmin=-max_abs_re, vmax=max_abs_re, filename='fig_un_re.jpg')
    max_abs_im = float(max(abs(numpy.min(numpy.imag(u))), abs(numpy.max(numpy.imag(u)))))
    myimshow(numpy.imag(u), title='$\operatorname{Im}(u_{n})$ in $\Omega$', colorbar='colorbar', cmap='jet', vmin=-max_abs_im, vmax=max_abs_im, filename='fig_un_im.jpg')
    chi_vmin = float(numpy.min(chi))
    chi_vmax = float(numpy.max(chi))
    myimshow(chi, title='$\chi_{n}$ in $\Omega$', colorbar='colorbar', cmap='jet', vmin=chi_vmin, vmax=chi_vmax, filename='fig_chin_re.jpg')
    # k_begin = 0
    # k_end = len(x) - 1
    # for k in range(k_begin, k_end):
    #     x_plot[k] = k
    #     y_plot[k] = chi[int(y[k]), int(x[k])]
    # matplotlib.pyplot.plot(x_plot, y_plot)
    # matplotlib.pyplot.title('$\chi_{n}$ in $\Omega$')
    # matplotlib.pyplot.show()

    return


def _plot_error(err):

    max_abs_re = float(max(abs(numpy.min(numpy.real(err))), abs(numpy.max(numpy.real(err)))))
    myimshow(numpy.real(err), title='$\operatorname{Re}(u_{n}-u_{0})$ in $\Omega$', colorbar='colorbar', cmap='jet', vmin=-max_abs_re, vmax=max_abs_re, filename='fig_err_real.jpg')
    max_abs_im = float(max(abs(numpy.min(numpy.imag(err))), abs(numpy.max(numpy.imag(err)))))
    myimshow(numpy.imag(err), title='$\operatorname{Im}(u_{n}-u_{0})$ in $\Omega$', colorbar='colorbar', cmap='jet', vmin=-max_abs_im, vmax=max_abs_im, filename='fig_err.jpg')

    return

def _plot_abs_u_history(u,chi):
    abs_u=numpy.abs(u)
    abs_vmin = float(numpy.min(abs_u))
    abs_vmax = float(numpy.max(abs_u))
    myimshow(abs_u, title='$\operatorname{abs}(u_{0})$ in $\Omega$', colorbar='colorbar', cmap='jet', vmin=abs_vmin, vmax=abs_vmax, filename='fig_u0_abs.jpg')
    chi_vmin = float(numpy.min(chi))
    chi_vmax = float(numpy.max(chi))
    myimshow(chi, title='$\chi_{0}$ in $\Omega$', colorbar='colorbar', cmap='jet', vmin=chi_vmin, vmax=chi_vmax, filename='fig_chi0_re.jpg')
    return

def _plot_energy_history(energy):

    matplotlib.pyplot.plot(energy) #, cmap = 'jet')#, vmin = 1e-4, vmax = 1e-0)
    matplotlib.pyplot.title('Energy')
    #matplotlib.pyplot.colorbar()
    #matplotlib.pyplot.show()
    filename = 'fig_energy_real.jpg'
    matplotlib.pyplot.savefig(filename)
    matplotlib.pyplot.close()

    return
