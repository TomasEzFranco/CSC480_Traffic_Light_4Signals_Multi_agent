# -----------------------------
# Before you use this app
# -----------------------------
# - RNG_SEED will affect the rng of the simulator, different values means different spawn rates etc.
# - simulationTime is how long the sim will run until it exits
#       and files will show up in the directory.
# - The files made are just graphs of the throughput and queue's (number of cars in a lane at once).
# - The graphs generated are dummy files that we can use to just show:
#       "These are what a fixed-time agent and a reflexive agent will look like"
# - Thats pretty much it. Run the file, see the graphs, then let's move onto adding RL and NN agents

import random
import time
import threading
import pygame
import sys
import matplotlib.pyplot as plt

# -----------------------------
# Reproducibility
# -----------------------------
RNG_SEED = 123
random.seed(RNG_SEED)

# -----------------------------
# Simulation settings
# -----------------------------
simulationTime = 60   # seconds

# Two-phase controller settings
MIN_GREEN = 8         # minimum seconds green must stay before allowing a switch
YELLOW_TIME = 3       # yellow duration (seconds)

# Vehicle spawn rate (seconds per vehicle)
SPAWN_EVERY = 1

# Vehicles speeds
speeds = {'car': 2.25, 'bus': 1.8, 'truck': 1.8, 'bike': 2.5}

# Allowed vehicle types
allowedVehicleTypes = {'car': True, 'bus': True, 'truck': True, 'bike': True}

# Directions
# 0: right, 1: down, 2: left, 3: up
directionNumbers = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}

# Phases:
# phase 0 = NS green (down, up)
# phase 1 = EW green (right, left)
PHASE_NS = 0
PHASE_EW = 1
PHASE_DIRS = {
    PHASE_NS: {'down', 'up'},
    PHASE_EW: {'right', 'left'}
}

# Lane anchor coordinates (do NOT mutate these)
lane_x = {'right': [0, 0, 0], 'down': [755, 727, 697], 'left': [1400, 1400, 1400], 'up': [602, 627, 657]}
lane_y = {'right': [348, 370, 398], 'down': [0, 0, 0], 'left': [498, 466, 436], 'up': [800, 800, 800]}

# Vehicles storage
vehicles = {
    'right': {0: [], 1: [], 2: [], 'crossed': 0},
    'down':  {0: [], 1: [], 2: [], 'crossed': 0},
    'left':  {0: [], 1: [], 2: [], 'crossed': 0},
    'up':    {0: [], 1: [], 2: [], 'crossed': 0}
}

# After crossing, keep straight-moving vehicles for spacing
vehiclesNotTurned = {'right': {1: [], 2: []}, 'down': {1: [], 2: []}, 'left': {1: [], 2: []}, 'up': {1: [], 2: []}}

# Stop lines & default stop points
stopLines = {'right': 590, 'down': 330, 'left': 800, 'up': 535}
defaultStop = {'right': 580, 'down': 320, 'left': 810, 'up': 545}

# Gaps
stoppingGap = 25
movingGap = 25

# UI coordinates
signalCoods = [(530, 230), (810, 230), (810, 570), (530, 570)]
signalTimerCoods = [(530, 210), (810, 210), (810, 550), (530, 550)]
timeElapsedCoods = (1100, 50)
vehicleCountCoods = [(480, 210), (880, 210), (880, 550), (480, 550)]

# Screen
screenWidth = 1400
screenHeight = 800

pygame.init()
simulation = pygame.sprite.Group()

# Global flag
stop_simulation = False

# -----------------------------
# Spawn helpers (fixed spawn positions)
# -----------------------------
SPAWN_MARGIN = 120  # how far off-screen to spawn
IMAGE_CACHE = {}

def get_vehicle_image(direction, vehicleClass):
    key = (direction, vehicleClass)
    if key not in IMAGE_CACHE:
        IMAGE_CACHE[key] = pygame.image.load(f"images/{direction}/{vehicleClass}.png")
    return IMAGE_CACHE[key]

def spawn_position(direction, lane):
    if direction == "right":
        return -SPAWN_MARGIN, lane_y["right"][lane]
    if direction == "left":
        return screenWidth + SPAWN_MARGIN, lane_y["left"][lane]
    if direction == "down":
        return lane_x["down"][lane], -SPAWN_MARGIN
    return lane_x["up"][lane], screenHeight + SPAWN_MARGIN

def can_spawn(direction, lane, vehicleClass):
    img = get_vehicle_image(direction, vehicleClass)
    w = img.get_rect().width
    h = img.get_rect().height
    sx, sy = spawn_position(direction, lane)
    lane_list = vehicles[direction][lane]
    if not lane_list:
        return True
    last = lane_list[-1]
    lw = last.image.get_rect().width
    lh = last.image.get_rect().height
    if direction == "right":
        return last.x >= sx + w + stoppingGap
    if direction == "left":
        return sx >= last.x + lw + stoppingGap
    if direction == "down":
        return last.y >= sy + h + stoppingGap
    return sy >= last.y + lh + stoppingGap

# -----------------------------
# Metrics & Graphing
# -----------------------------
metrics = {
    "t": [],
    "ns_q": [],
    "ew_q": [],
    "total_q": [],
    "phase": [],
    "in_yellow": [],
    "throughput": [],     # vehicles removed (offscreen) per second
    "switch_req": [],     # agent requested switch (1/0)
}
vehicles_removed_this_second = 0

# -----------------------------
# Signal controller state (two-phase)
# -----------------------------
phase = PHASE_NS
in_yellow = False
yellow_remaining = 0
green_elapsed = 0

# time elapsed
timeElapsed = 0

def getAllowedVehicleTypeNames():
    return [name for name, enabled in allowedVehicleTypes.items() if enabled]

# -----------------------------
# Sim timer thread
# -----------------------------
def simTime():
    global timeElapsed, stop_simulation
    while True:
        time.sleep(1)
        timeElapsed += 1
        if timeElapsed >= simulationTime:
            stop_simulation = True
            return

def plot_metrics(title="Run"):
    t = metrics["t"]
    plt.figure()
    plt.plot(t, metrics["ns_q"], label="NS queue")
    plt.plot(t, metrics["ew_q"], label="EW queue")
    plt.plot(t, metrics["total_q"], label="Total queue")
    plt.xlabel("time (s)")
    plt.ylabel("vehicles waiting")
    plt.title(title + " - Queues")
    plt.legend()
    plt.tight_layout()
    plt.savefig("queues.png")

    plt.figure()
    plt.plot(t, metrics["throughput"], label="Throughput (veh/s)")
    plt.xlabel("time (s)")
    plt.ylabel("vehicles exited screen")
    plt.title(title + " - Throughput")
    plt.legend()
    plt.tight_layout()
    plt.savefig("throughput.png")

    plt.figure()
    plt.plot(t, metrics["phase"], label="Phase (0=NS, 1=EW)")
    plt.plot(t, metrics["in_yellow"], label="In yellow")
    plt.xlabel("time (s)")
    plt.ylabel("state")
    plt.title(title + " - Signal State")
    plt.legend()
    plt.tight_layout()
    plt.savefig("signal_state.png")

# -----------------------------
# Agents
# -----------------------------
class Agent:
    def act(self, queues, phase, in_yellow, green_elapsed):
        """Return True to request a switch (controller enforces MIN_GREEN)."""
        return False

class ReflexQueueAgent(Agent):
    def __init__(self, threshold=6, max_hold=25):
        self.threshold = threshold
        self.max_hold = max_hold

    def act(self, queues, phase, in_yellow, green_elapsed):
        if in_yellow:
            return False
        ns = queues['down'] + queues['up']
        ew = queues['right'] + queues['left']
        if phase == PHASE_NS:
            if ew - ns >= self.threshold:
                return True
            if green_elapsed >= self.max_hold and ew > 0:
                return True
        else:
            if ns - ew >= self.threshold:
                return True
            if green_elapsed >= self.max_hold and ns > 0:
                return True
        return False

class FixedTimeAgent(Agent):
    """
    Pretimed controller: hold green for fixed duration (fixed_green), then request switch.
    Ignores queues (good baseline).
    """
    def __init__(self, fixed_green=15):
        self.fixed_green = fixed_green
    def act(self, queues, phase, in_yellow, green_elapsed):
        if in_yellow:
            return False
        return green_elapsed >= self.fixed_green

class MaxPressureAgent(Agent):
    """
    Simple max-pressure: compute pressure of each phase and request switch when opposing
    pressure exceeds current pressure + margin.
    Pressure = sum(queues of approaches that would have green) - sum(queues of approaches that WOULD be red)
    We use a simple margin to avoid flip-flopping.
    """
    def __init__(self, margin=2):
        self.margin = margin

    def phase_pressure(self, queues, phase_val):
        # pressure for phase_val: sum queues that would be green under phase_val
        if phase_val == PHASE_NS:
            return queues['down'] + queues['up']
        else:
            return queues['right'] + queues['left']

    def act(self, queues, phase, in_yellow, green_elapsed):
        if in_yellow:
            return False
        current_pressure = self.phase_pressure(queues, phase)
        opp_pressure = self.phase_pressure(queues, PHASE_EW if phase == PHASE_NS else PHASE_NS)
        # request if opposing pressure significantly higher
        return (opp_pressure - current_pressure) >= self.margin
    
class WaitTimeAgent(Agent):
    """
    Fairness-oriented agent:
    - Switch if the opposing phase has a much larger MAX waiting time than the current phase.
    - Also switch if current green has been held too long and opponent has anyone waiting.

    This prevents starvation even if one approach has a small queue.
    """
    def __init__(self, max_wait_margin=5, max_hold=30):
        self.max_wait_margin = max_wait_margin
        self.max_hold = max_hold

    def act(self, queues, phase, in_yellow, green_elapsed):
        if in_yellow:
            return False

        max_wait, avg_wait = get_wait_stats()

        ns_max = max(max_wait['down'], max_wait['up'])
        ew_max = max(max_wait['right'], max_wait['left'])

        ns_q = queues['down'] + queues['up']
        ew_q = queues['right'] + queues['left']

        if phase == PHASE_NS:
            # NS is green; switch if EW has much older waiting vehicles
            if (ew_max - ns_max) >= self.max_wait_margin and ew_q > 0:
                return True
            if green_elapsed >= self.max_hold and ew_q > 0:
                return True
        else:
            # EW is green; switch if NS has much older waiting vehicles
            if (ns_max - ew_max) >= self.max_wait_margin and ns_q > 0:
                return True
            if green_elapsed >= self.max_hold and ns_q > 0:
                return True

        return False


# pick agent type here: "reflex", "fixed", "pressure", "wait"
AGENT_TYPE = "reflex"
if AGENT_TYPE == "fixed":
    agent = FixedTimeAgent(fixed_green=12)
elif AGENT_TYPE == "reflex":
    agent = ReflexQueueAgent(threshold=6, max_hold=25)
elif AGENT_TYPE == "pressure":
    agent = MaxPressureAgent(margin=2)
else:
    agent = WaitTimeAgent(max_wait_margin=5, max_hold=30)

# -----------------------------
# Controller functions
# -----------------------------
def is_green_for(direction_name):
    return (not in_yellow) and (direction_name in PHASE_DIRS[phase])

def is_yellow_for(direction_name):
    return in_yellow and (direction_name in PHASE_DIRS[phase])

def signal_step(request_switch):
    global phase, in_yellow, yellow_remaining, green_elapsed
    if in_yellow:
        yellow_remaining -= 1
        if yellow_remaining <= 0:
            phase = PHASE_EW if phase == PHASE_NS else PHASE_NS
            in_yellow = False
            green_elapsed = 0
        return
    # green state
    green_elapsed += 1
    # enforce minimum green
    if request_switch and green_elapsed >= MIN_GREEN:
        in_yellow = True
        yellow_remaining = YELLOW_TIME

def get_queues():
    WAIT_DIST = 80
    q = {'right': 0, 'down': 0, 'left': 0, 'up': 0}
    for d in q.keys():
        for lane in (1, 2):
            lane_list = vehicles[d][lane]
            for idx, v in enumerate(lane_list):
                if v.crossed != 0:
                    continue
                ahead_clear = True
                if d == 'right':
                    front = v.x + v.image.get_rect().width
                    close_to_stop = front >= (stopLines[d] - WAIT_DIST)
                    if idx > 0:
                        prev = lane_list[idx - 1]
                        ahead_clear = (front < (prev.x - movingGap))
                elif d == 'left':
                    front = v.x
                    close_to_stop = front <= (stopLines[d] + WAIT_DIST)
                    if idx > 0:
                        prev = lane_list[idx - 1]
                        ahead_clear = (front > (prev.x + prev.image.get_rect().width + movingGap))
                elif d == 'down':
                    front = v.y + v.image.get_rect().height
                    close_to_stop = front >= (stopLines[d] - WAIT_DIST)
                    if idx > 0:
                        prev = lane_list[idx - 1]
                        ahead_clear = (front < (prev.y - movingGap))
                else:  # up
                    front = v.y
                    close_to_stop = front <= (stopLines[d] + WAIT_DIST)
                    if idx > 0:
                        prev = lane_list[idx - 1]
                        ahead_clear = (front > (prev.y + prev.image.get_rect().height + movingGap))
                if close_to_stop or (not ahead_clear):
                    q[d] += 1
    return q


def get_wait_stats():
    """
    Returns:
      max_wait: dict(direction -> max wait seconds among vehicles not yet crossed)
      avg_wait: dict(direction -> avg wait seconds among vehicles not yet crossed)
    """
    max_wait = {'right': 0, 'down': 0, 'left': 0, 'up': 0}
    avg_wait = {'right': 0, 'down': 0, 'left': 0, 'up': 0}

    for d in max_wait.keys():
        waits = []
        for lane in (1, 2):
            for v in vehicles[d][lane]:
                if v.crossed == 0:
                    waits.append(timeElapsed - getattr(v, "spawn_time", timeElapsed))
        if waits:
            max_wait[d] = max(waits)
            avg_wait[d] = sum(waits) / len(waits)

    return max_wait, avg_wait


# -----------------------------
# Vehicles
# -----------------------------
class Vehicle(pygame.sprite.Sprite):
    def __init__(self, lane, vehicleClass, direction_number, direction):
        super().__init__()
        self.lane = lane
        self.vehicleClass = vehicleClass
        self.speed = speeds[vehicleClass]
        self.direction_number = direction_number
        self.direction = direction
        self.spawn_time = timeElapsed

        # use cached image + fixed spawn pos
        self.image = get_vehicle_image(direction, vehicleClass)
        self.x, self.y = spawn_position(direction, lane)

        self.crossed = 0
        self.crossedIndex = 0

        vehicles[direction][lane].append(self)
        self.index = len(vehicles[direction][lane]) - 1

        # stopping position based on vehicle ahead (if it hasn't crossed)
        if len(vehicles[direction][lane]) > 1 and vehicles[direction][lane][self.index - 1].crossed == 0:
            prev = vehicles[direction][lane][self.index - 1]
            if direction == 'right':
                self.stop = prev.stop - prev.image.get_rect().width - stoppingGap
            elif direction == 'left':
                self.stop = prev.stop + prev.image.get_rect().width + stoppingGap
            elif direction == 'down':
                self.stop = prev.stop - prev.image.get_rect().height - stoppingGap
            else:  # up
                self.stop = prev.stop + prev.image.get_rect().height + stoppingGap
        else:
            self.stop = defaultStop[direction]

        simulation.add(self)

    def _remove_if_offscreen(self):
        global vehicles_removed_this_second
        margin = 150
        if (self.x > screenWidth + margin or self.x < -margin or
                self.y > screenHeight + margin or self.y < -margin):
            vehicles_removed_this_second += 1
            lane_list = vehicles[self.direction][self.lane]
            try:
                pos = lane_list.index(self)
            except ValueError:
                pos = None
            if pos is not None:
                lane_list.pop(pos)
                for i in range(pos, len(lane_list)):
                    lane_list[i].index = i
            try:
                nt_list = vehiclesNotTurned[self.direction].get(self.lane, [])
                if self in nt_list:
                    idx = nt_list.index(self)
                    nt_list.pop(idx)
                    for i in range(idx, len(nt_list)):
                        nt_list[i].crossedIndex = i
            except Exception:
                pass
            self.kill()

    def update_stop(self):
        """
        Recompute stop position every frame so vehicles roll forward to the stop line
        (or behind the vehicle in front) instead of using stale spawn-time stop values.
        """
        lane_list = vehicles[self.direction][self.lane]

        # If I'm the first vehicle in this lane (or the vehicle ahead already crossed),
        # my stop should be the default stop line.
        if self.index == 0:
            self.stop = defaultStop[self.direction]
            return

        prev = lane_list[self.index - 1]
        if prev.crossed != 0:
            self.stop = defaultStop[self.direction]
            return

        # Otherwise, stop behind the *current position* of the vehicle ahead.
        if self.direction == "right":
            # stop is a FRONT-x threshold (x + width <= stop)
            self.stop = min(defaultStop["right"], prev.x - stoppingGap)

        elif self.direction == "left":
            # stop is a LEFT-x threshold (x >= stop)
            self.stop = max(defaultStop["left"], prev.x + prev.image.get_rect().width + stoppingGap)

        elif self.direction == "down":
            # stop is a FRONT-y threshold (y + height <= stop)
            self.stop = min(defaultStop["down"], prev.y - stoppingGap)

        else:  # "up"
            # stop is a TOP-y threshold (y >= stop)
            self.stop = max(defaultStop["up"], prev.y + prev.image.get_rect().height + stoppingGap)

    def move(self):
        # keep stop target fresh while approaching the intersection
        if self.crossed == 0:
            self.update_stop()

            if self.direction == 'right' and (self.x + self.image.get_rect().width) > stopLines[self.direction]:
                self.crossed = 1
            elif self.direction == 'down' and (self.y + self.image.get_rect().height) > stopLines[self.direction]:
                self.crossed = 1
            elif self.direction == 'left' and self.x < stopLines[self.direction]:
                self.crossed = 1
            elif self.direction == 'up' and self.y < stopLines[self.direction]:
                self.crossed = 1
            if self.crossed == 1:
                vehicles[self.direction]['crossed'] += 1
                if self.lane in (1, 2):
                    vehiclesNotTurned[self.direction][self.lane].append(self)
                    self.crossedIndex = len(vehiclesNotTurned[self.direction][self.lane]) - 1

        # movement rules: obey signal before crossing, spacing always
        if self.direction == 'right':
            if self.crossed == 0:
                canGoSignal = is_green_for('right')
                frontClear = (self.index == 0 or (self.x + self.image.get_rect().width) < (vehicles[self.direction][self.lane][self.index - 1].x - movingGap))
                if ((self.x + self.image.get_rect().width) <= self.stop or canGoSignal) and frontClear:
                    self.x += self.speed
            else:
                frontClear = (self.crossedIndex == 0 or (self.x + self.image.get_rect().width) < (vehiclesNotTurned[self.direction][self.lane][self.crossedIndex - 1].x - movingGap))
                if frontClear:
                    self.x += self.speed

        elif self.direction == 'down':
            if self.crossed == 0:
                canGoSignal = is_green_for('down')
                frontClear = (self.index == 0 or (self.y + self.image.get_rect().height) < (vehicles[self.direction][self.lane][self.index - 1].y - movingGap))
                if ((self.y + self.image.get_rect().height) <= self.stop or canGoSignal) and frontClear:
                    self.y += self.speed
            else:
                frontClear = (self.crossedIndex == 0 or (self.y + self.image.get_rect().height) < (vehiclesNotTurned[self.direction][self.lane][self.crossedIndex - 1].y - movingGap))
                if frontClear:
                    self.y += self.speed

        elif self.direction == 'left':
            if self.crossed == 0:
                canGoSignal = is_green_for('left')
                frontClear = (self.index == 0 or self.x > (vehicles[self.direction][self.lane][self.index - 1].x + vehicles[self.direction][self.lane][self.index - 1].image.get_rect().width + movingGap))
                if (self.x >= self.stop or canGoSignal) and frontClear:
                    self.x -= self.speed
            else:
                frontClear = (self.crossedIndex == 0 or self.x > (vehiclesNotTurned[self.direction][self.lane][self.crossedIndex - 1].x + vehiclesNotTurned[self.direction][self.lane][self.crossedIndex - 1].image.get_rect().width + movingGap))
                if frontClear:
                    self.x -= self.speed

        elif self.direction == 'up':
            if self.crossed == 0:
                canGoSignal = is_green_for('up')
                frontClear = (self.index == 0 or self.y > (vehicles[self.direction][self.lane][self.index - 1].y + vehicles[self.direction][self.lane][self.index - 1].image.get_rect().height + movingGap))
                if (self.y >= self.stop or canGoSignal) and frontClear:
                    self.y -= self.speed
            else:
                frontClear = (self.crossedIndex == 0 or self.y > (vehiclesNotTurned[self.direction][self.lane][self.crossedIndex - 1].y + vehiclesNotTurned[self.direction][self.lane][self.crossedIndex - 1].image.get_rect().height + movingGap))
                if frontClear:
                    self.y -= self.speed

        self._remove_if_offscreen()

# -----------------------------
# Vehicle generator thread
# -----------------------------
def generateVehicles():
    allowedTypeNames = getAllowedVehicleTypeNames()
    if not allowedTypeNames:
        raise ValueError("No vehicle types enabled in allowedVehicleTypes")
    while not stop_simulation:
        spawned = False
        # Try a few times to find a lane with room
        for _ in range(10):
            vehicleClass = random.choice(allowedTypeNames)
            lane_number = random.randint(1, 2)
            temp = random.randint(0, 99)
            dist = [25, 50, 75, 100]
            if temp < dist[0]:
                direction_number = 0
            elif temp < dist[1]:
                direction_number = 1
            elif temp < dist[2]:
                direction_number = 2
            else:
                direction_number = 3
            direction = directionNumbers[direction_number]
            if can_spawn(direction, lane_number, vehicleClass):
                Vehicle(lane_number, vehicleClass, direction_number, direction)
                spawned = True
                break
        time.sleep(SPAWN_EVERY)

def showStats():
    totalVehicles = 0
    print('Direction-wise Vehicle Counts')
    for i in range(0, 4):
        d = directionNumbers[i]
        print('Direction', i + 1, ':', vehicles[d]['crossed'])
        totalVehicles += vehicles[d]['crossed']
    print('Total vehicles crossed stoplines:', totalVehicles)
    print('Total time:', timeElapsed)
    s = summarize()
    print("\nSummary:")
    for k, v in s.items():
        print(f"  {k}: {v}")

def summarize():
    total_q = metrics["total_q"]
    thr = metrics["throughput"]
    switches = sum(metrics["switch_req"]) if metrics["switch_req"] else 0
    total_thr = sum(thr) if thr else 0
    return {
        "avg_queue": (sum(total_q) / len(total_q)) if total_q else 0,
        "max_queue": max(total_q) if total_q else 0,
        "total_throughput": total_thr,
        "switch_requests": switches,
        "throughput_per_switch_req": (total_thr / switches) if switches else float("inf"),
    }

# -----------------------------
# Main
# -----------------------------
def main():
    global vehicles_removed_this_second
    black = (0, 0, 0)
    white = (255, 255, 255)
    screen = pygame.display.set_mode((screenWidth, screenHeight))
    pygame.display.set_caption("SIMULATION - Two-Phase + Agent")
    background = pygame.image.load('images/intersection.png')
    redSignal = pygame.image.load('images/signals/red.png')
    yellowSignal = pygame.image.load('images/signals/yellow.png')
    greenSignal = pygame.image.load('images/signals/green.png')
    font = pygame.font.Font(None, 30)
    threading.Thread(name="generateVehicles", target=generateVehicles, daemon=True).start()
    threading.Thread(name="simTime", target=simTime, daemon=True).start()
    clock = pygame.time.Clock()
    decision_accum = 0.0
    while True:
        dt = clock.tick(60) / 1000.0
        decision_accum += dt
        if stop_simulation:
            showStats()
            plot_metrics(title=agent.__class__.__name__)
            pygame.quit()
            sys.exit()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                showStats()
                plot_metrics(title=agent.__class__.__name__)
                sys.exit()
        # 1-second controller update + metrics
        while decision_accum >= 1.0:
            decision_accum -= 1.0
            q = get_queues()
            request_switch = agent.act(q, phase, in_yellow, green_elapsed)
            signal_step(request_switch)
            ns_q = q['down'] + q['up']
            ew_q = q['right'] + q['left']
            total_q = ns_q + ew_q
            t_now = len(metrics["t"])
            metrics["t"].append(t_now)
            metrics["ns_q"].append(ns_q)
            metrics["ew_q"].append(ew_q)
            metrics["total_q"].append(total_q)
            metrics["phase"].append(phase)
            metrics["in_yellow"].append(int(in_yellow))
            metrics["switch_req"].append(int(request_switch))
            metrics["throughput"].append(vehicles_removed_this_second)
            vehicles_removed_this_second = 0
        # Rendering
        screen.blit(background, (0, 0))
        for i in range(4):
            dname = directionNumbers[i]
            if is_green_for(dname):
                screen.blit(greenSignal, signalCoods[i])
                sig_text = "G"
            elif is_yellow_for(dname):
                screen.blit(yellowSignal, signalCoods[i])
                sig_text = str(yellow_remaining)
            else:
                screen.blit(redSignal, signalCoods[i])
                sig_text = "R"
            txt = font.render(sig_text, True, white, black)
            screen.blit(txt, signalTimerCoods[i])
        for i in range(4):
            dname = directionNumbers[i]
            displayText = vehicles[dname]['crossed']
            countSurf = font.render(str(displayText), True, black, white)
            screen.blit(countSurf, vehicleCountCoods[i])
        timeElapsedText = font.render(("Time Elapsed: " + str(timeElapsed)), True, black, white)
        screen.blit(timeElapsedText, timeElapsedCoods)
        for v in simulation:
            screen.blit(v.image, [v.x, v.y])
            v.move()
        pygame.display.update()

if __name__ == "__main__":
    main()