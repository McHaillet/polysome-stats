import numpy as np

# or switch to scipy.spatial.transform.Rotation

epsilon = 1E-8


class Vector:
    # Class can be used as both a 3d coordinate, and a vector
    # TODO SCIPY probably also has a vector class
    def __init__(self, coordinates, normalize=False):
        """
        Init vector with (x,y,z) coordinates, assumes (0,0,0) origin.
        """
        assert len(coordinates) == 3, 'Invalid axis list for a 3d vector, input does not contain 3 coordinates.'
        self._axis = np.array(coordinates)
        self._zero_vector = np.all(self._axis==0)
        if normalize:
            self.normalize()

    def get(self):
        """
        Return vector in numpy array.
        """
        return self._axis

    def show(self):
        """
        Print the vector.
        """
        print(self._axis)

    def copy(self):
        """
        Return a copy of the vector (also class Vector).
        """
        return Vector(self.get())

    def inverse(self):
        """
        Inverse the vector (in place).
        """
        return Vector(self._axis * -1)

    def cross(self, other):
        """
        Get cross product of self and other Vector. Return as new vector.
        """
        return Vector([self._axis[1] * other._axis[2] - self._axis[2] * other._axis[1],
                       self._axis[2] * other._axis[0] - self._axis[0] * other._axis[2],
                       self._axis[0] * other._axis[1] - self._axis[1] * other._axis[0]])

    def dot(self, other):
        """
        Return the dot product of vectors v1 and v2, of form (x,y,z).
        Dot product of two vectors is zero if they are perpendicular.
        """
        return self._axis[0] * other._axis[0] + self._axis[1] * other._axis[1] + self._axis[2] * other._axis[2]

    def magnitude(self):
        """
        Calculate the magnitude (length) of vector p.
        """
        return np.sqrt(np.sum(self._axis ** 2))

    def normalize(self):
        """
        Normalize self by dividing by magnitude.
        """
        if not self._zero_vector:
            self._axis = self._axis / self.magnitude()

    def angle(self, other, degrees=False):
        """
        Get angle between self and other.
        """
        # returns angle in radians
        if self._zero_vector or other._zero_vector:
            angle = 0
        else:
            angle = np.arccos(self.dot(other) / (self.magnitude() * other.magnitude()))
        if degrees:
            return angle * 180 / np.pi
        else:
            return angle

    def rotate(self, rotation_matrix):
        """
        Rotate the vector in place by the rotation matrix.
        """
        return Vector(np.dot(self._axis, rotation_matrix))

    def _get_orthogonal_unit_vector(self):
        """
        Get some orthogonal unit vector, multiple solutions are possible. Private method used in get rotation.
        """
        # A vector orthogonal to (a, b, c) is (-b, a, 0), or (-c, 0, a) or (0, -c, b).
        if self._zero_vector:
            return Vector([1, 0, 0])  # zero vector is orthogonal to any vector
        else:
            if self._axis[2] != 0:
                x, y = 1, 1
                z = (- 1 / self._axis[2]) * (x * self._axis[0] + y * self._axis[1])
            elif self._axis[1] != 0:
                x, z = 1, 1
                y = (- 1 / self._axis[1]) * (x * self._axis[0] + z * self._axis[2])
            else:
                y, z = 1, 1
                x = (- 1 / self._axis[0]) * (y * self._axis[1] + z * self._axis[2])
            orth = Vector([x, y, z])
            orth.normalize()
            np.testing.assert_allclose(self.dot(orth), 0, atol=1e-7, err_msg='something went wrong in finding ' \
                                                                             'perpendicular vector')
            return orth

    def get_rotation(self, other):
        """
        Get rotation to rotate other vector onto self. Take the transpose of result to rotate self onto other.
        """
        if self._zero_vector or other._zero_vector:
            return np.identity(3)

        nself, nother = self.copy(), other.copy()
        nself.normalize()
        nother.normalize()

        if nself.dot(nother) > 0.99999:  # if the vectors are parallel
            return np.identity(3)  # return identity
        elif nself.dot(nother) < -0.99999:  # if the vectors are opposite
            axis = nself._get_orthogonal_unit_vector()  # return 180 along whatever axis
            angle = np.pi  # 180 degrees rotation around perpendicular vector
        else:
            axis = nself.cross(nother)
            axis.normalize()
            angle = nself.angle(nother)

        x, y, z = axis.get()
        c = np.cos(angle)
        s = np.sin(angle)
        t = 1.0 - c

        m00 = c + x * x * t
        m11 = c + y * y * t
        m22 = c + z * z * t

        tmp1 = x * y * t
        tmp2 = z * s
        m10 = tmp1 + tmp2
        m01 = tmp1 - tmp2
        tmp1 = x * z * t
        tmp2 = y * s
        m20 = tmp1 - tmp2
        m02 = tmp1 + tmp2
        tmp1 = y * z * t
        tmp2 = x * s
        m21 = tmp1 + tmp2
        m12 = tmp1 - tmp2

        mat = np.array([[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]])

        return mat


def rotation_matrix_x(angle):
    """Return the 3x3 rotation matrix around x axis.

    @param angle: rotation angle around x axis (in degree).

    @return: rotation matrix.
    """
    angle = np.deg2rad(angle)
    mtx = np.zeros((3, 3))
    mtx[1, 1] = np.cos(angle)
    mtx[2, 1] = np.sin(angle)
    mtx[2, 2] = np.cos(angle)
    mtx[1, 2] = -np.sin(angle)
    mtx[0, 0] = 1

    return mtx


def rotation_matrix_y(angle):
    """Return the 3x3 rotation matrix around y axis.

    @param angle: rotation angle around y axis (in degree).

    @return: rotation matrix.
    """
    angle = np.deg2rad(angle)
    mtx = np.zeros((3, 3))
    mtx[0, 0] = np.cos(angle)
    mtx[2, 0] = -np.sin(angle)
    mtx[2, 2] = np.cos(angle)
    mtx[0, 2] = np.sin(angle)
    mtx[1, 1] = 1

    return mtx


def rotation_matrix_z(angle):
    """Return the 3x3 rotation matrix around z axis.

    @param angle: rotation angle around z axis (in degree).

    @return: rotation matrix.
    """
    angle = np.deg2rad(angle)
    mtx = np.zeros((3, 3))
    mtx[0, 0] = np.cos(angle)
    mtx[1, 0] = np.sin(angle)
    mtx[1, 1] = np.cos(angle)
    mtx[0, 1] = -np.sin(angle)
    mtx[2, 2] = 1

    return mtx


def rotation_matrix_zxz(zxz):
    """Return the 3x3 rotation matrix of an Euler angle in ZXZ convention.
    Note the order of the specified angle should be [Phi, Theta, Psi], or [Z1, X, Z2].
    Rotation matrix multiplied in order mat(Z2) * mat(X) * mat(Z1).

    @param angle: list of [Phi, Theta, Psi] in degree.

    @return: rotation matrix.
    """
    assert len(zxz) == 3

    z1, x, z2 = zxz

    zm1 = rotation_matrix_z(z1)
    xm = rotation_matrix_x(x)
    zm2 = rotation_matrix_z(z2)

    res = np.dot(zm2, np.dot(xm, zm1))

    return res


def rotation_matrix_zyz(zyz):
    """Return the 3x3 rotation matrix of an Euler angle in ZYZ convention.
    Note the order of the specified angle should be [Phi, Theta, Psi], or [Z1, X, Z2].
    Rotation matrix multiplied in order mat(Z2) * mat(X) * mat(Z1)

    @param angle: list of [Phi, Theta, Psi] in degree.

    @return: rotation matrix.
    """
    assert len(zyz) == 3

    z1, y, z2 = zyz

    zm1 = rotation_matrix_z(z1)
    xm = rotation_matrix_y(y)
    zm2 = rotation_matrix_z(z2)

    res = np.dot(zm2, np.dot(xm, zm1))

    return res


def mat2xyz(r):
    y = np.rad2deg(np.arcsin(r[0, 2]))

    if np.abs(90 - y) < epsilon or np.abs(90 + y) < epsilon:
        z = np.rad2deg(np.arctan2(r[1,0], r[1,1]))
        x = 0
    else:
        x = np.rad2deg(np.arctan2(-r[1, 2], r[2, 2]))
        z = np.rad2deg(np.arctan2(-r[0, 1], r[0, 0]))

    return (x, y, z)


def mat2xzy(r):
    z = np.rad2deg(np.arcsin(-r[0, 1]))

    if np.abs(90 - z) < epsilon or np.abs(90 + z) < epsilon:
        y = np.rad2deg(np.arctan2(-r[2, 0], r[2, 2]))
        x = 0
    else:
        x = np.rad2deg(np.arctan2(r[2, 1], r[1, 1]))
        y = np.rad2deg(np.arctan2(r[0, 2], r[0, 0]))

    return (x, z, y)


def mat2yxz(r):
    x = np.rad2deg(np.arcsin(-r[1, 2]))

    if np.abs(90 - x) < epsilon or np.abs(90 + x) < epsilon:
        z = np.rad2deg(np.arctan2(-r[0, 1], r[0,0]))
        y = 0
    else:
        y = np.rad2deg(np.arctan2(r[0, 2], r[2, 2]))
        z = np.rad2deg(np.arctan2(r[1, 0], r[1, 1]))

    return (y, x, z)


def mat2yzx(r):
    z = np.rad2deg(np.arcsin(r[1, 0]))

    if np.abs(90- z) < epsilon or np.abs(90 + z) < epsilon:
        x = np.rad2deg(np.arctan2(r[2, 1], r[2, 2]))
        y = 0
    else:
        y = np.rad2deg(np.arctan2(-r[2, 0], r[0, 0]))
        x = np.rad2deg(np.arctan2(-r[1, 2], r[1, 1]))

    return (y, z, x)


def mat2zxy(r):
    x = np.rad2deg(np.arcsin(r[2, 1]))

    if np.abs(90 - x) < epsilon or np.abs(90 + x) < epsilon:
        y = np.rad2deg(np.arctan2(r[0, 2], r[0, 0]))
        z = 0
    else:
        z = np.rad2deg(np.arctan2(-r[0, 1], r[1, 1]))
        y = np.rad2deg(np.arctan2(-r[2, 0], r[2, 2]))

    return (z, x, y)


def mat2zyx(r):
    y = np.rad2deg(np.arcsin(-r[2, 0]))

    if np.abs(90 - y) < epsilon or np.abs(90 + y) < epsilon:
        x = np.rad2deg(np.arctan2(-r[1, 2], r[1, 1]))
        z = 0
    else:
        z = np.rad2deg(np.arctan2(r[1, 0], r[0, 0]))
        x = np.rad2deg(np.arctan2(r[2, 1], r[2, 2]))

    return (z, y, x)


def mat2xyx(r):
    y = np.rad2deg(np.arccos(r[0, 0]))

    if np.abs(180 - y) < epsilon or np.abs(y) < epsilon:
        x1 = np.rad2deg(np.arctan2(-r[1, 2], r[1, 1]))
        x0 = 0
    else:
        x0 = np.rad2deg(np.arctan2(r[1, 0], -r[2, 0]))
        x1 = np.rad2deg(np.arctan2(r[0, 1], r[0, 2]))

    return (x0, y, x1)


def mat2xzx(r):
    z = np.rad2deg(np.arccos(r[0, 0]))

    if np.abs(180 - z) < epsilon or np.abs(z) < epsilon:
        x1 = np.rad2deg(np.arctan2(r[2, 1], r[2, 2]))
        x0 = 0
    else:
        x0 = np.rad2deg(np.arctan2(r[2, 0], r[1, 0]))
        x1 = np.rad2deg(np.arctan2(r[0, 2], -r[0, 1]))

    return (x0, z, x1)


def mat2yxy(r):
    x = np.rad2deg(np.arccos(r[1, 1]))

    if np.abs(180 - x) < epsilon or np.abs(x) < epsilon:
        y1 = np.rad2deg(np.arctan2(r[0, 2], r[0, 0]))
        y0 = 0
    else:
        y0 = np.rad2deg(np.arctan2(r[0, 1], r[2, 1]))
        y1 = np.rad2deg(np.arctan2(r[1, 0], -r[1, 2]))

    return (y0, x, y1)


def mat2yzy(r):
    z = np.rad2deg(np.arccos(r[1, 1]))

    if np.abs(180 - z) < epsilon or np.abs(z) < epsilon:
        y1 = np.rad2deg(np.arctan2(-r[2, 0], r[2, 2]))
        y0 = 0
    else:
        y0 = np.rad2deg(np.arctan2(r[2, 1], -r[0, 1]))
        y1 = np.rad2deg(np.arctan2(r[1, 2], r[1, 0]))

    return (y0, z, y1)


def mat2zxz(r):
    x = np.rad2deg(np.arccos(r[2,2]))

    if np.abs(180 - x) < epsilon or np.abs(x) < epsilon:
        z1 = np.rad2deg(np.arctan2(-r[0,1], r[0,0]))
        z0 = 0
    else:
        z0 = np.rad2deg(np.arctan2(r[0,2], -r[1,2]))
        z1 = np.rad2deg(np.arctan2(r[2,0], r[2,1]))

    return (z0, x, z1)


def mat2zyz(r):
    y = np.rad2deg(np.arccos(r[2,2]))

    if np.abs(180 - y) < epsilon or np.abs(y) < epsilon:
        z1 = np.rad2deg(np.arctan2(r[1,0], r[1,1]))
        z0 = 0
    else:
        z0 = np.rad2deg(np.arctan2(r[1,2], r[0,2]))
        z1 = np.rad2deg(np.arctan2(r[2,1], -r[2,0]))

    return (z0, y, z1)


def rotation_matrix(angles, rotation_order='zxz', multiplication='post'):

    assert len(angles) == 3, "should provide 3 angles"
    assert multiplication in ['pre', 'post'], "multiplication can only be pre or post"

    mat_dict = {'x': rotation_matrix_x, 'y': rotation_matrix_y, 'z': rotation_matrix_z}

    mtxs = []
    for angle, rot in zip(angles, rotation_order):
        mtxs.append(mat_dict[rot](-angle) if multiplication == 'post' else mat_dict[rot](angle))

    if multiplication == 'post':
        return np.dot(np.dot(mtxs[0], mtxs[1]), mtxs[2])
    else:
        return np.dot(mtxs[2], np.dot(mtxs[1], mtxs[0]))


def mat2ord(rotation_matrix, return_order='zyz', multiplication='post'):
    assert multiplication in ['pre', 'post'], "multiplication can only be pre or post"
    assert len(rotation_matrix.shape) == 2 and all([s == 3 for s in rotation_matrix.shape]), \
        "invalid rotation matrix shape"

    return_funcs = {'xyz': mat2xyz, 'xzy': mat2xzy, 'yxz': mat2yxz, 'yzx': mat2yzx, 'zxy': mat2zxy, 'zyx': mat2zyx,
                    'xyx': mat2xyx, 'xzx': mat2xzx, 'yxy': mat2yxy, 'yzy': mat2yzy, 'zxz': mat2zxz, 'zyz': mat2zyz}

    # if 'pre' multiplication, invert the matrix
    res = return_funcs[return_order](np.linalg.inv(rotation_matrix)) if \
        multiplication == 'pre' else return_funcs[return_order](rotation_matrix)

    # always take negative of angles
    return tuple([-r for r in res])  # if multiplication == 'post' else res


def convert_angles(angles, rotation_order='zxz', return_order='zyz', multiplication='post'):
    # get the rotation matrix with the input order
    m = rotation_matrix(angles, rotation_order=rotation_order, multiplication=multiplication)
    # get the angles with the specified output order
    return mat2ord(m, return_order=return_order, multiplication=multiplication)

