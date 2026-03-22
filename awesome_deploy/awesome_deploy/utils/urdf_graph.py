"""URDF graph traversal helpers used to derive stable joint/link orderings."""

import xml.etree.ElementTree as ET
from collections import defaultdict, deque


class UrdfGraph:
    """Builds a traversable robot graph from a URDF file.

    The deployment pipeline relies on multiple orderings:

    - file order, matching the order defined in the URDF;
    - BFS order, matching the ordering used by some simulators and datasets;
    - DFS order, occasionally useful for debugging tree structure.
    """

    def __init__(self, urdf_path):
        """Parses the URDF and precomputes parent/child adjacency maps."""
        self.urdf_path = urdf_path
        self.tree = ET.parse(urdf_path)
        self.root = self.tree.getroot()
        self._parent_to_children_joints = defaultdict(list)
        self._parent_to_children_links = defaultdict(list)
        self._child_links = set()
        self._parent_links = set()
        self._joints = []
        self._build_graph()

    def _build_graph(self):
        """Collects non-fixed joints and corresponding link connectivity."""
        for joint in self.root.findall("joint"):
            name = joint.get("name")
            jtype = joint.get("type")
            if jtype == "fixed":
                continue
            parent = joint.find("parent").get("link")
            child = joint.find("child").get("link")
            self._joints.append((name, parent, child, jtype))
            self._parent_to_children_joints[parent].append((child, name))
            self._parent_to_children_links[parent].append(child)
            self._child_links.add(child)
            self._parent_links.add(parent)

    def root_link(self):
        """Returns the inferred root link of the robot tree."""
        roots = list(self._parent_links - self._child_links)
        if roots:
            return roots[0]
        return self._joints[0][1] if self._joints else None

    def joint_order_by_file(self):
        """Returns movable joint names in the order they appear in the URDF."""
        order = []
        for joint in self.root.findall("joint"):
            name = joint.get("name")
            jtype = joint.get("type")
            if jtype in ("fixed", "floating"):
                continue
            order.append(name)
        return order

    def link_order_by_file(self):
        """Returns link names in file order."""
        order = []
        for link in self.root.findall("link"):
            name = link.get("name")
            if name is None:
                continue
            order.append(name)
        return order

    def bfs_joint_order(self):
        """Returns movable joint names in breadth-first tree order."""
        root_link = self.root_link()
        if root_link is None:
            return []
        order = []
        queue = deque([root_link])
        while queue:
            link = queue.popleft()
            for child_link, joint_name in self._parent_to_children_joints.get(link, []):
                order.append(joint_name)
                queue.append(child_link)
        return order

    def dfs_joint_order(self):
        """Returns movable joint names in depth-first tree order."""
        root_link = self.root_link()
        if root_link is None:
            return []
        order = []
        stack = [root_link]
        while stack:
            link = stack.pop()
            children = self._parent_to_children_joints.get(link, [])
            for child_link, joint_name in reversed(children):
                order.append(joint_name)
                stack.append(child_link)
        return order

    def bfs_link_order(self):
        """Returns link names in breadth-first tree order."""
        root_link = self.root_link()
        if root_link is None:
            return []
        order = []
        queue = deque([root_link])
        while queue:
            link = queue.popleft()
            order.append(link)
            for child in self._parent_to_children_links.get(link, []):
                queue.append(child)
        return order
