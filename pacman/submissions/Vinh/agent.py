
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
    Ghost (Hider) Agent - né Pacman greedy multi-step tối đa có thể.;

    Chiến lược:
    - GẦN Pacman (<=6): dùng Minimax với:
        + Ghost MAXIMIZE an toàn (khoảng cách + lối thoát).
        + Pacman được mô phỏng theo đúng chiến lược greedy (multi-step).
        + Bất kỳ state nào có Manhattan <= 1 là THUA (score cực âm).
    - XA Pacman (>6): dùng BFS tìm vùng xa & nhiều lối thoát, nhưng dùng
      vị trí Pacman DỰ ĐOÁN chứ không phải vị trí hiện tại.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Học dần speed thật của Pacman
        self.estimated_pacman_speed = 1
        # Luật: Manhattan <= 1 coi như thua
        self.lose_distance = 1

        # Vùng nguy hiểm theo khoảng cách
        self.close_danger_radius = 6
        self.medium_danger_radius = 10

        # Lưu vị trí Pacman bước trước để ước lượng speed
        self._last_enemy_position = None

    # ================== STEP CHÍNH ==================

    def step(
        self,
        map_state: np.ndarray,
        my_position: tuple,
        enemy_position: tuple,
        step_number: int,
    ) -> Move:

        # Cập nhật ước lượng pacman_speed từ chuyển động thực tế
        self._update_pacman_speed(enemy_position)

        distance_to_pacman = self._calculate_manhattan_distance(
            my_position, enemy_position
        )

        # Vùng nguy hiểm : Nếu gần Pacman (<=6) → dùng Minimax để chọn nước đi tốt nhất
        if distance_to_pacman <= self.close_danger_radius:
            # Depth lớn hơn khi cực gần
            if distance_to_pacman <= 3:
                depth = 4
            else:
                depth = 3

            score, best_move = self._minimax(
                my_pos=my_position,
                enemy_pos=enemy_position,
                depth=depth,
                is_ghost_turn=True,
                map_state=map_state,
            )

            # Nếu minimax trả về move hợp lệ -> dùng
            if best_move is not None and self._is_valid_move(
                my_position, best_move, map_state
            ):
                return best_move

            # Fallback nếu minimax thất bại
            return self._greedy_escape(my_position, enemy_position, map_state)

        # Vùng an toàn : Pacman XA (> 6 cells) → Dùng PREDICTIVE BFS
        else:
            # Bước 1 - Dự đoán Pacman sẽ ở đâu (multi-step)
            predicted_enemy_pos = self._predict_enemy_position(
                enemy_position, my_position, map_state
            )

            # Bước 2 - Tìm vị trí xa predicted position
            safe_positions = self._find_safe_positions(
                map_state, my_position, predicted_enemy_pos, top_k=5
            )

            if not safe_positions:
                return self._greedy_escape(
                    my_position, enemy_position, map_state
                )

            # Bước 3 - Chọn vị trí có nhiều lối thoát nhất (giữa các vị trí xa)
            best_position = max(
                safe_positions,
                key=lambda pos: self._count_escape_routes(pos, map_state),
            )

            # Bước 4 - Di chuyển đến vị trí đó
            next_move = self._get_next_move(
                my_position, best_position, map_state
            )

            return next_move if next_move else Move.STAY

    # ================== MODEL PACMAN ==================

    def _update_pacman_speed(self, current_enemy_pos: tuple):
        """
        Mỗi bước, từ 2 vị trí Pacman liên tiếp, ước lượng pacman_speed
        (distance chính là số ô Pacman đi vì Pacman đi thẳng 1 hướng).
        """
        if self._last_enemy_position is not None:
            dist = self._calculate_manhattan_distance(
                self._last_enemy_position, current_enemy_pos
            )
            if dist > self.estimated_pacman_speed:
                # Giới hạn nhẹ cho an toàn (không quá to)
                self.estimated_pacman_speed = min(dist, 4)
        self._last_enemy_position = current_enemy_pos

    def _predict_enemy_position(
        self, enemy_pos: tuple, my_pos: tuple, map_state: np.ndarray
    ) -> tuple:
        """
        Mô phỏng Pacman sẽ đi đâu bước tới theo chiến lược greedy multi-step,
        giống với PacmanAgent:
        - Tính row_diff, col_diff = ghost - pacman
        - Ưu tiên di chuyển theo trục giúp lại gần Ghost
        - desired_steps = abs(row_diff/col_diff)
        - Giới hạn số bước bởi estimated_pacman_speed
        """

        pr, pc = enemy_pos
        gr, gc = my_pos

        row_diff = gr - pr
        col_diff = gc - pc

        preferred_moves = []

        # Ưu tiên dọc
        if row_diff > 0:
            preferred_moves.append(Move.DOWN)
        elif row_diff < 0:
            preferred_moves.append(Move.UP)

        # Sau đó ngang
        if col_diff > 0:
            preferred_moves.append(Move.RIGHT)
        elif col_diff < 0:
            preferred_moves.append(Move.LEFT)

        # Thử các hướng ưu tiên (giống PacmanAgent)
        for move in preferred_moves:
            desired_steps = self._desired_steps_for_pacman(
                move, row_diff, col_diff
            )
            steps = self._max_valid_steps_for_pacman(
                enemy_pos, move, map_state, desired_steps
            )
            if steps > 0:
                return self._apply_steps(enemy_pos, move, steps)

        # Fallback: thử 4 hướng, chọn hướng giảm khoảng cách nhiều nhất
        best_pos = enemy_pos
        best_dist = self._calculate_manhattan_distance(enemy_pos, my_pos)

        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            steps = self._max_valid_steps_for_pacman(
                enemy_pos, move, map_state, self.estimated_pacman_speed
            )
            if steps <= 0:
                continue
            candidate = self._apply_steps(enemy_pos, move, steps)
            d = self._calculate_manhattan_distance(candidate, my_pos)
            if d < best_dist:
                best_dist = d
                best_pos = candidate

        return best_pos

    def _desired_steps_for_pacman(
        self, move: Move, row_diff: int, col_diff: int
    ) -> int:
        """Giống PacmanAgent._desired_steps."""
        if move in (Move.UP, Move.DOWN):
            return abs(row_diff)
        if move in (Move.LEFT, Move.RIGHT):
            return abs(col_diff)
        return 1

    def _max_valid_steps_for_pacman(
        self,
        pos: tuple,
        move: Move,
        map_state: np.ndarray,
        desired_steps: int,
    ) -> int:
        """
        Giống PacmanAgent._max_valid_steps nhưng dùng estimated_pacman_speed.
        """
        steps = 0
        max_steps = min(self.estimated_pacman_speed, max(1, desired_steps))
        current = pos
        for _ in range(max_steps):
            dr, dc = move.value
            next_pos = (current[0] + dr, current[1] + dc)
            if not self._is_valid_position(next_pos, map_state):
                break
            steps += 1
            current = next_pos
        return steps

    def _apply_steps(self, pos: tuple, move: Move, steps: int) -> tuple:
        dr, dc = move.value
        return (pos[0] + dr * steps, pos[1] + dc * steps)

    # ================== MINIMAX ==================

    def _evaluate_state(self, my_pos: tuple, enemy_pos: tuple, map_state):
        """
        Hàm heuristic đánh giá state cho Ghost:
        - Bị bắt (dist <= 1) → -1e6
        - Ưu tiên:
            + Khoảng cách lớn
            + Nhiều lối thoát
        - Phạt:
            + dist = 2 hoặc 3 (siêu nguy hiểm)
            + Ngõ cụt (escape_routes <= 1)
            + Cùng hàng/cột khi khoảng cách chưa đủ xa
        """
        dist = self._calculate_manhattan_distance(my_pos, enemy_pos)

        if dist <= self.lose_distance:
            return -1_000_000

        escape_routes = self._count_escape_routes(my_pos, map_state)
        dead_end = escape_routes <= 1

        score = 0.0

        # An toàn gần: dist = 2 hoặc 3 vẫn cực kỳ nguy hiểm
        if dist == 2:
            score -= 800
        elif dist == 3:
            score -= 300

        # Khoảng cách càng xa càng tốt
        score += 50.0 * dist

        # Lối thoát nhiều thì tốt
        score += 40.0 * escape_routes

        # Ngõ cụt rất xấu, đặc biệt khi Pacman không xa
        if dead_end:
            score -= 200.0
            if dist <= self.medium_danger_radius:
                score -= 200.0

        # Tránh đứng cùng hàng/cột với Pacman khi còn tương đối gần
        same_row_or_col = my_pos[0] == enemy_pos[0] or my_pos[1] == enemy_pos[1]
        if same_row_or_col and dist <= (self.estimated_pacman_speed * 3 + 2):
            score -= 150.0

        return score

    def _minimax(
        self,
        my_pos: tuple,
        enemy_pos: tuple,
        depth: int,
        is_ghost_turn: bool,
        map_state: np.ndarray,
    ):
        """
        Minimax cho Ghost.

        Ghost (MAX) = tránh bị bắt, tăng khoảng cách + lối thoát.
        Pacman (MIN) = được mô phỏng theo chiến lược greedy multi-step
                       nên lượt Pacman là DETERMINISTIC (không nhánh).
        """

        dist = self._calculate_manhattan_distance(my_pos, enemy_pos)

        # THUA theo luật: khoảng cách <= 1
        if dist <= self.lose_distance:
            return -1_000_000, Move.STAY

        # Hết depth → đánh giá heuristic
        if depth == 0:
            return self._evaluate_state(my_pos, enemy_pos, map_state), Move.STAY

        if is_ghost_turn:
            best_score = -float("inf")
            best_move = None

            # Cho Ghost thử 4 hướng + STAY
            candidate_moves = [
                Move.STAY,
                Move.UP,
                Move.DOWN,
                Move.LEFT,
                Move.RIGHT,
            ]

            for move in candidate_moves:
                if move == Move.STAY:
                    new_pos = my_pos
                else:
                    dr, dc = move.value
                    new_pos = (my_pos[0] + dr, my_pos[1] + dc)

                if move != Move.STAY and not self._is_valid_position(
                    new_pos, map_state
                ):
                    continue

                score, _ = self._minimax(
                    new_pos,
                    enemy_pos,
                    depth - 1,
                    False,  # tới lượt Pacman
                    map_state,
                )

                if score > best_score:
                    best_score = score
                    best_move = move

            if best_move is None:
                # Không có nước đi hợp lệ → đứng yên, đánh giá luôn
                return (
                    self._evaluate_state(my_pos, enemy_pos, map_state),
                    Move.STAY,
                )

            return best_score, best_move

        else:
            # Lượt Pacman: chỉ MỘT state kế tiếp, mô phỏng greedy multi-step
            next_enemy_pos = self._predict_enemy_position(
                enemy_pos, my_pos, map_state
            )
            score, _ = self._minimax(
                my_pos,
                next_enemy_pos,
                depth - 1,
                True,
                map_state,
            )
            return score, Move.STAY

    # ================== HELPER GỐC (giữ & dùng lại) ==================

    def _is_valid_move(
        self, pos: tuple, move: Move, map_state: np.ndarray
    ) -> bool:
        delta_row, delta_col = move.value
        new_pos = (pos[0] + delta_row, pos[1] + delta_col)
        return self._is_valid_position(new_pos, map_state)

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        row, col = pos
        height, width = map_state.shape

        if row < 0 or row >= height or col < 0 or col >= width:
            return False

        return map_state[row, col] == 0

    def _calculate_manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _find_safe_positions(self, map_state, my_pos, enemy_pos, top_k=5):
        visited = set()
        queue = deque([(my_pos, 0)])
        visited.add(my_pos)
        positions_with_distance = []

        while queue:
            current_pos, dist_from_start = queue.popleft()

            dist_to_enemy = self._calculate_manhattan_distance(
                current_pos, enemy_pos
            )
            positions_with_distance.append((current_pos, dist_to_enemy))

            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                new_pos = (current_pos[0] + dr, current_pos[1] + dc)

                if new_pos not in visited and self._is_valid_position(
                    new_pos, map_state
                ):
                    visited.add(new_pos)
                    queue.append((new_pos, dist_from_start + 1))

        positions_with_distance.sort(key=lambda x: x[1], reverse=True)
        return [pos for pos, dist in positions_with_distance[:top_k]]

    def _count_escape_routes(self, pos, map_state):
        count = 0
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            neighbor_pos = (pos[0] + dr, pos[1] + dc)
            if self._is_valid_position(neighbor_pos, map_state):
                count += 1
        return count

    def _get_next_move(self, start, goal, map_state):
        if start == goal:
            return Move.STAY

        visited = {start}
        queue = deque([(start, [])])

        while queue:
            pos, path = queue.popleft()

            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                new_pos = (pos[0] + dr, pos[1] + dc)

                if new_pos == goal and self._is_valid_position(
                    new_pos, map_state
                ):
                    if path:
                        return path[0]
                    else:
                        return move

                if new_pos not in visited and self._is_valid_position(
                    new_pos, map_state
                ):
                    visited.add(new_pos)
                    new_path = path + [move]
                    queue.append((new_pos, new_path))

        return Move.STAY

    def _greedy_escape(self, my_pos, enemy_pos, map_state):
        row_diff = my_pos[0] - enemy_pos[0]
        col_diff = my_pos[1] - enemy_pos[1]

        if abs(row_diff) > abs(col_diff):
            move = Move.DOWN if row_diff > 0 else Move.UP
        else:
            move = Move.RIGHT if col_diff > 0 else Move.LEFT

        if self._is_valid_move(my_pos, move, map_state):
            return move

        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_move(my_pos, move, map_state):
                return move

        return Move.STAY
