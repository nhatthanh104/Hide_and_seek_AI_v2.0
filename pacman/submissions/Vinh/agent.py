
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
    Ghost (Hider) Agent - Tối ưu né Pac-Man với chiến lược hybrid (heuristic + minimax đánh giá thời gian sống sót).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(
        self,
        map_state: np.ndarray,
        my_position: tuple,
        enemy_position: tuple,
        step_number: int,
    ) -> Move:
        from collections import deque

        gx, gy = my_position
        px, py = enemy_position
        rows, cols = map_state.shape

        def is_valid(pos):
            x, y = pos
            return 0 <= x < rows and 0 <= y < cols and map_state[x][y] == 0

        def in_line_of_sight(gx, gy, px, py):
            if gx == px:
                step = 1 if py > gy else -1
                for y in range(gy + step, py, step):
                    if not is_valid((gx, y)):
                        return False
                return True
            elif gy == py:
                step = 1 if px > gx else -1
                for x in range(gx + step, px, step):
                    if not is_valid((x, gy)):
                        return False
                return True
            return False

        def bfs_distance(start, goal):
            if start == goal:
                return 0
            visited = [[False] * cols for _ in range(rows)]
            q = deque([(start[0], start[1], 0)])
            visited[start[0]][start[1]] = True
            while q:
                cx, cy, dist = q.popleft()
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = cx + dx, cy + dy
                    if (
                        0 <= nx < rows
                        and 0 <= ny < cols
                        and not visited[nx][ny]
                        and is_valid((nx, ny))
                    ):
                        if (nx, ny) == goal:
                            return dist + 1
                        visited[nx][ny] = True
                        q.append((nx, ny, dist + 1))
            return float("inf")

        def simulate_pacman(pos, direction, steps):
            x, y = pos
            for _ in range(steps):
                nx, ny = x + direction[0], y + direction[1]
                if not is_valid((nx, ny)):
                    break
                x, y = nx, ny
            return (x, y)

        def get_direction(from_pos, to_pos):
            dx = to_pos[0] - from_pos[0]
            dy = to_pos[1] - from_pos[1]
            if dx == 1:
                return Move.DOWN
            if dx == -1:
                return Move.UP
            if dy == 1:
                return Move.RIGHT
            if dy == -1:
                return Move.LEFT
            return Move.STAY

        # --- 1. Né đường thẳng nguy hiểm ---
        danger_dist = 2
        if (
            in_line_of_sight(gx, gy, px, py)
            and abs(gx - px) + abs(gy - py) <= danger_dist
        ):
            perp_moves = [(1, 0), (-1, 0)] if gx == px else [(0, 1), (0, -1)]
            for dx, dy in perp_moves:
                nx, ny = gx + dx, gy + dy
                if is_valid((nx, ny)):
                    return get_direction((gx, gy), (nx, ny))

        # --- 2. Né ngõ cụt ---
        valid_neighbors = [
            (gx + dx, gy + dy)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if is_valid((gx + dx, gy + dy))
        ]
        if len(valid_neighbors) == 1 and abs(gx - px) + abs(gy - py) <= 4:
            return get_direction((gx, gy), valid_neighbors[0])

        # --- 3. Maximin sống sót ---
        possible_moves = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
        best_move = (0, 0)
        max_survival = -float("inf")

        pac_dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        pacman_moves = [d for d in pac_dirs if is_valid((px + d[0], py + d[1]))]

        straight_move = None
        if step_number > 1 and hasattr(self, "prev_enemy_pos"):
            dx = px - self.prev_enemy_pos[0]
            dy = py - self.prev_enemy_pos[1]
            if (dx, dy) in pacman_moves:
                straight_move = (dx, dy)

        for dx, dy in possible_moves:
            nx, ny = gx + dx, gy + dy
            if not is_valid((nx, ny)):
                continue
            ghost_next = (nx, ny)
            worst_time = float("inf")
            for pdx, pdy in pacman_moves:
                steps = 2 if (pdx, pdy) == straight_move else 1
                pac_next = simulate_pacman((px, py), (pdx, pdy), steps)
                if pac_next == ghost_next:
                    time = 0
                else:
                    time = bfs_distance(ghost_next, pac_next)
                worst_time = min(worst_time, time)
            if worst_time > max_survival:
                max_survival = worst_time
                best_move = (dx, dy)

        self.prev_enemy_pos = enemy_position
        return get_direction((gx, gy), (gx + best_move[0], gy + best_move[1]))
