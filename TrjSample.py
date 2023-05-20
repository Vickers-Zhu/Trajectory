class TrjSample:
    def __init__(self, trajectory, frame, tag, pos_x, pos_y, pos_z, epos_x, epos_y, epos_z, spos_x, spos_y, distort, veloc_x, veloc_y, veloc_z, length, a_xi, a_eta, a_zeta):
        self.trajectory = trajectory
        self.frame = frame
        self.tag = tag
        self.pos = Position(pos_x, pos_y, pos_z)
        self.epos = Position(epos_x, epos_y, epos_z)
        self.spos = Position(spos_x, spos_y)
        self.distort = distort
        self.veloc = Velocity(veloc_x, veloc_y, veloc_z)
        self.len = length
        self.a_xi = a_xi
        self.a_eta = a_eta
        self.a_zeta = a_zeta


class Position:
    def __init__(self, x, y, z=None):
        self.x = x
        self.y = y
        self.z = z


class Velocity:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
