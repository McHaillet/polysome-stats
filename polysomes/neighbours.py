import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from rotation import Vector, rotation_matrix, mat2ord
from mpl_toolkits.axes_grid1 import make_axes_locatable


def neighbour_position_3d(dataframe, neighbourhood=5, plane_norm=[0, 0, 1], reference=[0, 0, 1],
                          estimate_membrane_plane=False, class_column_name='', center_classes=[], neighbour_classes=[]):

    tomograms = np.unique(dataframe.tomogram)
    classes = np.unique(dataframe[class_column_name]) if class_column_name != '' else []

    if len(center_classes) == 0 and len(classes) != 0:
        center_classes = np.unique(classes)
    if len(neighbour_classes) == 0 and len(classes) != 0:
        neighbour_classes = np.unique(classes)

    # plane adjustment
    reference = Vector(reference)
    plane_norm = Vector(plane_norm)
    rm_adjust = reference.get_rotation(plane_norm)

    # list of output coordinates
    relative_coords = []
    n_center_particles = 0

    for tomo in tomograms:
        # select tomogram from dataframe
        df_tomogram = dataframe[dataframe.tomogram == tomo]
        coordinates = np.array([df_tomogram.x, df_tomogram.y, df_tomogram.z]).T
        rotations = np.array([df_tomogram.rlnAngleRot, df_tomogram.rlnAngleTilt, df_tomogram.rlnAnglePsi]).T

        n_part_in_tomo = df_tomogram.shape[0]
        if n_part_in_tomo < 2:
            continue
        else:
            if len(classes) != 0:
                n_center_particles += len([c for c in df_tomogram[class_column_name] if c in center_classes])
            else:
                n_center_particles += df_tomogram.shape[0]

        # calculate distance matrix
        dist_matrix = cdist(coordinates, coordinates)
        max_dist = np.max(dist_matrix)
        dist_matrix[dist_matrix == 0] = max_dist + 1

        for i, (row, particle) in enumerate(df_tomogram.iterrows()):

            if len(classes) != 0 and not particle[class_column_name] in center_classes:
                continue

            loop = neighbourhood if (n_part_in_tomo - 1) >= neighbourhood else n_part_in_tomo - 1

            for _ in range(loop):

                j = np.argmin(dist_matrix[i])
                distance = dist_matrix[i][j]
                neighbour = df_tomogram.iloc[j]  # use iloc for relative index in new df
                dist_matrix[i][j] = max_dist + 1

                if len(classes) != 0 and not neighbour[class_column_name] in neighbour_classes:
                    continue

                if 100 < distance < 1000:
                    coord_p = coordinates[i]
                    coord_n = coordinates[j]

                    # vector from center particle to neighbour
                    v = Vector(coord_n - coord_p)

                    # voltools rotation matrix
                    rm = rotation_matrix(-rotations[i], rotation_order='zyz').T
                    vf = v.rotate(rm[:3, :3])

                    # append to results
                    relative_coords.append(list(vf.get()))

    relative_coords = np.array(relative_coords).T
    print('you have ', n_center_particles, ' particle centers')
    print('those have in total ', relative_coords[1].shape, ' neighbours considering ', neighbourhood,
          ' neighbours per particle')

    if estimate_membrane_plane:
        svd = np.linalg.svd(relative_coords[:, 0:25000] -
                            np.mean(relative_coords[:, 0:25000], axis=1, keepdims=True))
        left = svd[0]
        plane_norm = Vector(left[:, -1])
        rm_adjust = reference.get_rotation(plane_norm)

        print('estimated membrane plane vector ', left[:, -1])

    return np.dot(relative_coords.T, rm_adjust).T, plane_norm.get()


def neighbour_pos_and_rot(dataframe, neighbourhood=5, plane_norm=[0, 0, 1], reference=[0, 0, 1]):

    tomograms = np.unique(dataframe.tomogram)

    # plane adjustment
    reference = Vector(reference)
    plane_norm = Vector(plane_norm)
    rm_adjust = reference.get_rotation(plane_norm)

    # list of output coordinates
    relative_coords = []
    relative_rots = []
    n_center_particles = 0

    for tomo in tomograms:
        # select tomogram from dataframe
        df_tomogram = dataframe[dataframe.tomogram == tomo]
        coordinates = np.array([df_tomogram.x, df_tomogram.y, df_tomogram.z]).T
        rotations = np.array([df_tomogram.rlnAngleRot, df_tomogram.rlnAngleTilt, df_tomogram.rlnAnglePsi]).T

        n_part_in_tomo = df_tomogram.shape[0]
        if n_part_in_tomo < 2:
            continue
        else:
            n_center_particles += n_part_in_tomo

        # calculate distance matrix
        dist_matrix = cdist(coordinates, coordinates)
        max_dist = np.max(dist_matrix)
        dist_matrix[dist_matrix == 0] = max_dist + 1

        for i, particle in df_tomogram.iterrows():

            loop = neighbourhood if (n_part_in_tomo - 1) >= neighbourhood else n_part_in_tomo - 1

            for _ in range(loop):

                j = np.argmin(dist_matrix[i])
                distance = dist_matrix[i][j]
                dist_matrix[i][j] = max_dist + 1

                if 100 < distance < 1000:
                    coord_p = coordinates[i]
                    coord_n = coordinates[j]

                    # vector from center particle to neighbour
                    v = Vector(coord_n - coord_p)

                    # - voltools rotation matrix
                    # - tranpose changes rotation from ref-to-particle (relion convention) to particle-to-ref
                    rm = rotation_matrix(-rotations[i], rotation_order='zyz', multiplication='post').T
                    vf = v.rotate(rm[:3, :3]).rotate(rm_adjust)
                    relative_coords.append(list(vf.get()))  # append to results

                    # rotate neighbour rotations angles
                    rm_nb = rotation_matrix(-rotations[j], rotation_order='zyz', multiplication='post')
                    rm_nb_rot = np.dot(np.dot(rm[:3, :3], rm_adjust), rm_nb[:3, :3])
                    rel_rot = mat2ord(rm_nb_rot, return_order='zyz', multiplication='post')  # relative rotation of
                    # neighbour in the frame of the reference
                    relative_rots.append(rel_rot)

    relative_coords = np.array(relative_coords)
    relative_rots = np.array(relative_rots)
    return relative_coords, relative_rots


def find_leading_trailing(dataframe, trailing_mask, leading_mask, pixels_mask, neighbourhood=5,
                          plane_norm=[0, 0, 1], reference=[0, 0, 1], class_column_name='',
                          center_classes=[], neighbour_classes=[]):
    # set default classes if not provided
    tomograms = np.unique(dataframe.tomogram)
    classes = np.unique(dataframe[class_column_name]) if class_column_name != '' else []

    if len(center_classes) == 0 and len(classes) != 0:
        center_classes = np.unique(classes)
    if len(neighbour_classes) == 0 and len(classes) != 0:
        neighbour_classes = np.unique(classes)

    # add column for leading and trailing ids, filled with -1 by default
    if not ('trailing_id' in dataframe.columns and 'leading_id' in dataframe.columns):
        dataframe['trailing_id'] = [-1, ] * dataframe.shape[0]
        dataframe['leading_id'] = [-1, ] * dataframe.shape[0]
        # else the fields are already there, do not overwrite them

    # unpack the bin limits
    x_bins, y_bins, z_bins = pixels_mask
    # index bins with: x_loc = np.searchsorted(x_bins, rel_x_coordinate) - 1
    # ...
    # if trailing_mask[x_loc, y_loc, z_loc]:
    #     particle['trailing_id'] = neighbour.id

    # plane adjustment
    reference = Vector(reference)
    plane_norm = Vector(plane_norm)
    rm_adjust = reference.get_rotation(plane_norm)

    # list for relative coordinates
    relative_coords_poly, relative_coords_non_poly = [], []
    n_center_particles = 0

    for tomo_nr, tomo in enumerate(tomograms):

        # select tomogram from dataframe
        df_tomogram = dataframe[dataframe.tomogram == tomo]
        coordinates = np.array([df_tomogram.x, df_tomogram.y, df_tomogram.z]).T
        rotations = np.array([df_tomogram.rlnAngleRot, df_tomogram.rlnAngleTilt, df_tomogram.rlnAnglePsi]).T

        n_part_in_tomo = df_tomogram.shape[0]
        if n_part_in_tomo < 2:
            continue
        else:
            if len(classes) != 0:
                n_center_particles += len([c for c in df_tomogram[class_column_name] if c in center_classes])
            else:
                n_center_particles += df_tomogram.shape[0]

        dist_matrix = cdist(coordinates, coordinates)
        max_dist = np.max(dist_matrix)
        dist_matrix[dist_matrix == 0] = max_dist + 1

        # search for polysomes
        for i, (row, particle) in enumerate(df_tomogram.iterrows()):

            if len(classes) != 0 and not particle[class_column_name] in center_classes:
                continue

            loop = neighbourhood if (n_part_in_tomo - 1) >= neighbourhood else n_part_in_tomo - 1

            for _ in range(loop):

                j = np.argmin(dist_matrix[i])
                # distance = dist_matrix[i][j]
                neighbour = df_tomogram.iloc[j]  # use iloc for relative index in new df
                dist_matrix[i][j] = max_dist + 1

                if len(classes) != 0 and not neighbour[class_column_name] in neighbour_classes:
                    continue

                # get coordinates of current and neighbour
                coord_p = coordinates[i]
                coord_n = coordinates[j]

                # find neighbor position relative to center
                v = Vector(coord_n - coord_p)
                rm = rotation_matrix(-rotations[i], rotation_order='zyz', multiplication='post').T
                vf = v.rotate(rm[:3, :3]).rotate(rm_adjust)
                rel_coords = vf.get()

                # do the inverse for the center relative to neighbor
                vn = Vector(coord_p - coord_n)
                rmn = rotation_matrix(-rotations[j], rotation_order='zyz', multiplication='post').T
                vnf = vn.rotate(rmn[:3, :3]).rotate(rm_adjust)  # vector final
                rel_coords_n = vnf.get()

                # make sure the locations falls in the bounds of the mask
                if not ((x_bins.min() <= rel_coords[0] <= x_bins.max()) and
                        (y_bins.min() <= rel_coords[1] <= y_bins.max()) and
                        (z_bins.min() <= rel_coords[2] <= z_bins.max())):
                    continue
                if not ((x_bins.min() <= rel_coords_n[0] <= x_bins.max()) and
                        (y_bins.min() <= rel_coords_n[1] <= y_bins.max()) and
                        (z_bins.min() <= rel_coords_n[2] <= z_bins.max())):
                    continue

                x_loc = np.searchsorted(x_bins, rel_coords[0]) - 1
                y_loc = np.searchsorted(y_bins, rel_coords[1]) - 1
                z_loc = np.searchsorted(z_bins, rel_coords[2]) - 1

                # neighbour
                x_loc_n = np.searchsorted(x_bins, rel_coords_n[0]) - 1
                y_loc_n = np.searchsorted(y_bins, rel_coords_n[1]) - 1
                z_loc_n = np.searchsorted(z_bins, rel_coords_n[2]) - 1

                # fill only if:
                # - no neighbour has been assigned to the trailing/leading position
                # - and the neighbor falls in the masked region, and center in inverse region for neighbor
                if ((trailing_mask[x_loc, y_loc, z_loc] and leading_mask[x_loc_n, y_loc_n, z_loc_n]) and
                        (dataframe.loc[row, 'trailing_id'] == -1 and
                         dataframe.loc[neighbour.name, 'leading_id'] == -1)):

                    # record association
                    dataframe.loc[row, 'trailing_id'] = neighbour.name
                    dataframe.loc[neighbour.name, 'leading_id'] = row

                    # add coordinates
                    relative_coords_poly.append(rel_coords)
                    relative_coords_poly.append(rel_coords_n)

                elif ((leading_mask[x_loc, y_loc, z_loc] and trailing_mask[x_loc_n, y_loc_n, z_loc_n]) and
                      (dataframe.loc[row, 'leading_id'] == -1 and dataframe.loc[neighbour.name, 'trailing_id'] == -1)):

                    # record association
                    dataframe.loc[row, 'leading_id'] = neighbour.name
                    dataframe.loc[neighbour.name, 'trailing_id'] = row

                    # add coordinates
                    relative_coords_poly.append(rel_coords)
                    relative_coords_poly.append(rel_coords_n)

                else:  # add coordinates to the non-polysome list for later inspection
                    relative_coords_non_poly.append(rel_coords)

    relative_coords_poly = np.array(relative_coords_poly).T
    relative_coords_non_poly = np.array(relative_coords_non_poly).T

    return dataframe, relative_coords_poly, relative_coords_non_poly


def density_plot(data, hist_limits, hist_voxel_size, fig_size, vrange=None, probability=True, plane='xy',
                 tick_labels=None, colormap_plt='afmhot'):
    """
    @param data: (3, N) shape array, x, y, z coordinates with A units
    @param hist_limits: tuple of two elements giving min and max value for all axis
    @param hist_voxel_size: voxel size of histogram in A units
    """
    assert data.shape[0] == 3, 'data does not have x, y, z as the second axis'

    plane_to_axis = {'xy': 2,
                     'xz': 1,
                     'yz': 0}

    # calculate bins for the dimensions
    n_bins = int(abs(hist_limits[1] - hist_limits[0]) / hist_voxel_size)

    hist_3d, hist_3d_edges = np.histogramdd(data.T, bins=n_bins,
                                            range=[(hist_limits[0], hist_limits[1]), ] * 3, density=False)
    if probability:
        hist_3d /= hist_3d.sum()

    fig, ax = plt.subplots(figsize=fig_size)

    # imshow takes array (M,N) as rows, columns, tranpose to get x as columns because input is (X,Y)
    if vrange is not None:
        h = ax.imshow(hist_3d.sum(axis=plane_to_axis[plane]).T,
                      cmap=colormap_plt, vmin=vrange[0], vmax=vrange[1], origin='lower')
    else:
        h = ax.imshow(hist_3d.sum(axis=plane_to_axis[plane]).T,
                      cmap=colormap_plt, origin='lower')

    # add colorbar and axis
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(h, cax=cax)
    if probability:
        cbar.set_label('Probability')
    else:
        cbar.set_label('# particles')
    ax.set_xlabel(f'Relative {plane[0]}-coordinate $(\AA)$')
    ax.set_ylabel(f'Relative {plane[1]}-coordinate $(\AA)$')

    # ticks = [0, hist_3d.shape[0] // 2, hist_3d.shape[0] - 1]
    if tick_labels is None:
        tick_labels = [hist_limits[0], 0, hist_limits[1]]
    print('axis tick labels: ', tick_labels)
    tick_locs = np.round((((np.array(tick_labels) - hist_limits[0]) /
                           (hist_limits[-1] - hist_limits[0])) * n_bins)).astype(int)

    # print(tick_locs)
    ax.set_xticks(tick_locs)
    ax.set_xticklabels([str(l) for l in tick_labels])

    ax.set_yticks(tick_locs)
    ax.set_yticklabels([str(l) for l in tick_labels])

    return fig, ax, hist_3d, hist_3d_edges


def plot_3d(data, plane_norm):
    # ======= plot
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')

    if data.shape[1] > 20000:
        display = data[:, np.random.choice(data.shape[1], size=20000)]
    else:
        display = data

    ax.scatter(display[0], display[1], display[2], s=0.01)
    ax.quiver(0, 0, 0, *plane_norm, length=100, color='red', label='membrane plane')
    ax.quiver(0, 0, 0, 0, 0, 1, length=100, color='blue', label='plane correction')

    ax.set_xlabel('x')
    ax.set_xlim(-300, 300)
    ax.set_ylabel('y')
    ax.set_ylim(-300, 300)
    ax.set_zlabel('z')
    ax.set_zlim(-300, 300)
    return fig, ax

