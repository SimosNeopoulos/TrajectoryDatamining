class Point:

    def __init__(self, time, x, y, node_id, speed, next_node_x, next_node_y, next_node_id):
        self.time = float(time)
        self.x = int(x)
        self.y = int(y)
        self.node_id = int(node_id)
        self.speed = float(speed)
        self.next_node_x = int(next_node_x)
        self.next_node_y = int(next_node_y)
        self.next_node_id = int(next_node_id)

