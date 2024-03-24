from point import Point


class Trajectory:

    def __init__(self, object_id, object_class_id):
        self.id = int(object_id)
        self.object_class_id = int(object_class_id)
        self.trajectory_path = []
        self.edge_list = []

    def __eq__(self, other):
        return self.id == other.id

    def add_point(self, point: Point):
        self.trajectory_path.append(point)

    def get_path_list(self):
        return [str(point.node_id) for point in self.trajectory_path]

    def get_edge_list(self):
        if not self.edge_list:
            self._create_edge_list()

        return self.edge_list

    def _create_edge_list(self):
        prev_point = self.trajectory_path[0]
        for point in self.trajectory_path[1:]:
            self.edge_list.append((prev_point.node_id, point.next_node_id))
            prev_point = point

