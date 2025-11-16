
import heapq
import math
import sys
import time
from collections import deque
from pathlib import Path

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

import numpy as np
from agent_interface import GhostAgent as BaseGhostAgent
from agent_interface import PacmanAgent as BasePacmanAgent
from environment import Move

"""
Hợp lệ theo yêu cầu: chỉ dùng thư viện chuẩn + time.
"""


class PacmanAgent(BasePacmanAgent):
    """
    Tác nhân Pacman sử dụng thuật toán A* (A Star) để đuổi bắt Ghost.
    Có xử lý chống xuyên qua nhau (swap collision).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "A* Pacman Hunter"

    # =====================================================
    # HÀM CHÍNH: ĐƯỢC GỌI MỖI LƯỢT DI CHUYỂN
    # =====================================================
    def step(self, map_state, my_position, enemy_position, step_number):
        """
        Hàm quyết định hướng di chuyển mỗi bước.
        - Sử dụng thuật toán A* để tìm đường tối ưu đuổi theo Ghost.
        - Có giới hạn thời gian tối đa 1 giây cho mỗi bước.
        """

        start_time = (
            time.time()
        )  # Ghi nhận thời điểm bắt đầu để giới hạn thời gian

        # Tìm đường đi ngắn nhất bằng A*
        path = self.a_star_search(
            map_state, my_position, enemy_position, start_time
        )

        # Nếu tìm được đường đi hợp lệ (ít nhất 2 ô: hiện tại + bước kế tiếp)
        if path and len(path) > 1:
            next_pos = path[1]  # Lấy bước kế tiếp để di chuyển

            # =====================================================
            #  NGĂN XUYÊN QUA NHAU (SWAP COLLISION)
            # =====================================================
            # Dự đoán các ô mà Ghost có thể di chuyển đến trong lượt tiếp theo
            enemy_next_positions = []
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                new_enemy_pos = (enemy_position[0] + dr, enemy_position[1] + dc)
                if self._is_valid_position(new_enemy_pos, map_state):
                    enemy_next_positions.append(new_enemy_pos)

            # Nếu Pacman định đi vào đúng ô Ghost đang đứng
            # Hoặc Pacman và Ghost cùng đổi vị trí trong cùng một lượt
            if next_pos == enemy_position or (
                my_position in enemy_next_positions
                and enemy_position == next_pos
            ):
                # Khi đó Pacman sẽ đứng yên để tránh “xuyên qua nhau”
                return Move.STAY

            # Di chuyển theo hướng từ vị trí hiện tại đến vị trí kế tiếp
            return self._position_to_move(my_position, next_pos)

        # Nếu không tìm thấy đường hợp lệ trong 1s → dùng chiến lược đuổi "tham lam"
        return self._greedy_chase(map_state, my_position, enemy_position)

    # =====================================================
    # THUẬT TOÁN A* (A STAR SEARCH)
    # =====================================================
    def a_star_search(self, map_state, start, goal, start_time):
        """
        Thuật toán A*: tìm đường ngắn nhất từ start → goal.
        Có giới hạn thời gian thực thi là 1 giây.
        """

        # Hàm heuristic: khoảng cách Manhattan
        def manhattan_distance(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

        # Hàng đợi ưu tiên (priority queue)
        frontier = []
        heapq.heappush(frontier, (0, 0, start, [start]))  # (f, tie, node, path)
        explored = {start: 0}
        tie = 0  # Dùng để phá hòa khi f bằng nhau

        while frontier:
            # Giới hạn thời gian 1 giây để tránh vượt quá yêu cầu bài
            if time.time() - start_time > 1.0:
                break

            f, _, current, path = heapq.heappop(frontier)
            if current == goal:
                return path  # Đã đến đích

            g = len(path) - 1  # Chi phí thực tế từ start → current
            if explored.get(current, float("inf")) < g:
                continue

            # Duyệt 4 hướng có thể đi: lên, xuống, trái, phải
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                neighbor = (current[0] + dr, current[1] + dc)

                # Bỏ qua nếu vị trí không hợp lệ (ngoài bản đồ hoặc là tường)
                if not self._is_valid_position(neighbor, map_state):
                    continue

                new_g = g + 1
                # Nếu đã thăm với chi phí tốt hơn → bỏ qua
                if neighbor in explored and explored[neighbor] <= new_g:
                    continue

                explored[neighbor] = new_g
                h = manhattan_distance(neighbor, goal)
                tie += 1
                # Thêm vào hàng đợi với tổng chi phí f = g + h
                heapq.heappush(
                    frontier, (new_g + h, tie, neighbor, path + [neighbor])
                )

        # Nếu không tìm thấy đường hợp lệ → trả về rỗng
        return []

    # =====================================================
    # CÁC HÀM HỖ TRỢ
    # =====================================================
    def _greedy_chase(self, map_state, my_pos, enemy_pos):
        """
        Chiến lược đuổi đơn giản khi A* thất bại.
        Pacman sẽ đi theo hướng gần Ghost nhất (theo Manhattan distance).
        """
        row_diff = enemy_pos[0] - my_pos[0]
        col_diff = enemy_pos[1] - my_pos[1]
        moves = []

        # Ưu tiên hướng có khoảng cách lớn hơn
        if abs(row_diff) >= abs(col_diff):
            if row_diff > 0:
                moves.append(Move.DOWN)
            elif row_diff < 0:
                moves.append(Move.UP)
            if col_diff > 0:
                moves.append(Move.RIGHT)
            elif col_diff < 0:
                moves.append(Move.LEFT)
        else:
            if col_diff > 0:
                moves.append(Move.RIGHT)
            elif col_diff < 0:
                moves.append(Move.LEFT)
            if row_diff > 0:
                moves.append(Move.DOWN)
            elif row_diff < 0:
                moves.append(Move.UP)

        # Kiểm tra và chọn hướng di chuyển hợp lệ đầu tiên
        for move in moves + [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            new_pos = (my_pos[0] + dr, my_pos[1] + dc)
            if self._is_valid_position(new_pos, map_state):
                return move

        return Move.STAY  # Không có hướng hợp lệ thì đứng yên

    def _position_to_move(self, current, target):
        """Chuyển từ tọa độ đích sang hướng di chuyển (Move)."""
        dr = target[0] - current[0]
        dc = target[1] - current[1]
        if dr > 0:
            return Move.DOWN
        elif dr < 0:
            return Move.UP
        elif dc > 0:
            return Move.RIGHT
        elif dc < 0:
            return Move.LEFT
        return Move.STAY

    def _is_valid_position(self, pos, map_state):
        """Kiểm tra xem vị trí có hợp lệ không (trong bản đồ và không phải tường)."""
        r, c = pos
        if r < 0 or r >= len(map_state) or c < 0 or c >= len(map_state[0]):
            return False
        return map_state[r][c] == 0


class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) Agent - Planning evader:
    - Mỗi bước nhìn trước nhiều bước tương lai (depth-limited search)
    - Giả lập Pacman đuổi theo (greedy + speed 2)
    - Chọn move tối đa hóa số bước sống sót
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prev_pac_pos = None
        # độ sâu lookahead (có thể tune: 5, 6, 7...)
        self.search_depth = 6
        # giới hạn thời gian mỗi bước (giảm nếu bị chậm)
        self.time_limit = 0.03  # ~30ms

    # ----------------- tiện ích cơ bản -----------------

    def _is_valid(self, map_state, pos):
        x, y = pos
        rows, cols = map_state.shape
        return 0 <= x < rows and 0 <= y < cols and map_state[x, y] == 0

    def _neighbors4(self, map_state, pos):
        x, y = pos
        res = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if self._is_valid(map_state, (nx, ny)):
                res.append((nx, ny))
        return res

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _get_direction(self, from_pos, to_pos):
        fx, fy = from_pos
        tx, ty = to_pos
        dx, dy = tx - fx, ty - fy
        if dx == 1:
            return Move.DOWN
        if dx == -1:
            return Move.UP
        if dy == 1:
            return Move.RIGHT
        if dy == -1:
            return Move.LEFT
        return Move.STAY

    # ----------------- mô phỏng Pacman -----------------

    def _simulate_pacman(self, map_state, pac_pos, ghost_pos, prev_dir):
        """
        Giả lập Pacman:
        - Chọn hướng làm khoảng cách Manhattan tới ghost nhỏ nhất
        - Nếu prev_dir vẫn hợp lệ và tiếp tục giảm khoảng cách → ưu tiên đi tiếp
        - Đi thẳng tối đa 2 ô nếu không đụng tường
        """
        px, py = pac_pos

        # candidate directions
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        best_dir = None
        best_dist = float("inf")

        # thử tiếp tục đi theo prev_dir nếu có
        if prev_dir is not None:
            dx, dy = prev_dir
            nx, ny = px + dx, py + dy
            if self._is_valid(map_state, (nx, ny)):
                d = self._manhattan((nx, ny), ghost_pos)
                best_dir = (dx, dy)
                best_dist = d

        # nếu không có prev_dir tốt → chọn lại greedy
        for dx, dy in dirs:
            nx, ny = px + dx, py + dy
            if not self._is_valid(map_state, (nx, ny)):
                continue
            d = self._manhattan((nx, ny), ghost_pos)
            if d < best_dist:
                best_dist = d
                best_dir = (dx, dy)

        if best_dir is None:
            # Pacman bị kẹt
            return pac_pos, None

        # di 1–2 ô theo best_dir
        steps = 2  # speed 2
        x, y = px, py
        for _ in range(steps):
            nx, ny = x + best_dir[0], y + best_dir[1]
            if not self._is_valid(map_state, (nx, ny)):
                break
            x, y = nx, ny

        return (x, y), best_dir

    # ----------------- DFS planning -----------------

    def _plan_dfs(
        self,
        map_state,
        ghost_pos,
        pac_pos,
        depth,
        prev_pac_dir,
        start_time,
        cache,
    ):
        """
        Trả về: số bước sống sót kỳ vọng từ state này (kể cả step hiện tại trong DFS)
        Ghost luôn chọn move tối đa hóa kết quả.
        Pacman hành xử theo policy mô phỏng.
        """
        # cắt thời gian để tránh quá tải
        if time.time() - start_time > self.time_limit:
            # heuristic fallback: càng xa Pacman càng tốt
            return max(1, self._manhattan(ghost_pos, pac_pos))

        # nếu bị bắt rồi
        if self._manhattan(ghost_pos, pac_pos) <= 1:
            return 0

        if depth == 0:
            # sống được thêm 1 bước trong horizon
            return 1

        key = (ghost_pos, pac_pos, depth, prev_pac_dir)
        if key in cache:
            return cache[key]

        max_survival = 0

        # tất cả move ghost có thể đi
        moves = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
        for gdx, gdy in moves:
            gx, gy = ghost_pos
            ngx, ngy = gx + gdx, gy + gdy
            new_ghost = (ngx, ngy)

            if not self._is_valid(map_state, new_ghost):
                continue

            # Pacman di chuyển sau ghost
            new_pac, new_dir = self._simulate_pacman(
                map_state, pac_pos, new_ghost, prev_pac_dir
            )

            # nếu sau bước này bị bắt luôn
            if self._manhattan(new_ghost, new_pac) <= 1:
                survival = 1  # sống được 1 bước rồi chết
            else:
                survival = 1 + self._plan_dfs(
                    map_state,
                    new_ghost,
                    new_pac,
                    depth - 1,
                    new_dir,
                    start_time,
                    cache,
                )

            if survival > max_survival:
                max_survival = survival

        cache[key] = max_survival
        return max_survival

    # ----------------- step chính -----------------

    def step(
        self,
        map_state: np.ndarray,
        my_position: tuple,
        enemy_position: tuple,
        step_number: int,
    ) -> Move:

        start_time = time.time()
        cache = {}

        gx, gy = my_position
        px, py = enemy_position

        # đoán hướng Pacman bước trước (nếu có)
        prev_dir = None
        if self.prev_pac_pos is not None:
            dx = px - self.prev_pac_pos[0]
            dy = py - self.prev_pac_pos[1]
            if (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                prev_dir = (dx, dy)

        best_move_vec = (0, 0)
        best_value = -float("inf")

        # thử tất cả move có thể của ghost ở root
        moves = [
            (0, 0),
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ]

        for gdx, gdy in moves:
            ngx, ngy = gx + gdx, gy + gdy
            new_ghost = (ngx, ngy)
            if not self._is_valid(map_state, new_ghost):
                continue

            # Pacman move sau root-move này
            new_pac, new_dir = self._simulate_pacman(
                map_state, enemy_position, new_ghost, prev_dir
            )

            # nếu bị bắt ngay
            if self._manhattan(new_ghost, new_pac) <= 1:
                val = 1
            else:
                val = 1 + self._plan_dfs(
                    map_state,
                    new_ghost,
                    new_pac,
                    self.search_depth - 1,
                    new_dir,
                    start_time,
                    cache,
                )

            # tie-break: nếu val bằng nhau, chọn move tăng khoảng cách hơn
            if val > best_value:
                best_value = val
                best_move_vec = (gdx, gdy)
            elif val == best_value:
                # tie-break bằng khoảng cách Manhattan
                old_target = (gx + best_move_vec[0], gy + best_move_vec[1])
                if self._manhattan(new_ghost, enemy_position) > self._manhattan(
                    old_target, enemy_position
                ):
                    best_move_vec = (gdx, gdy)

        self.prev_pac_pos = enemy_position
        target_pos = (gx + best_move_vec[0], gy + best_move_vec[1])
        return self._get_direction(my_position, target_pos)
