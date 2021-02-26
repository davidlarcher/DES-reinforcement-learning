from __future__ import print_function  # compatibility with Python 2.x
from __future__ import division  # compatibility with Python 2.x

import random
import gym
from gym.spaces import Discrete, MultiDiscrete, Box
import numpy as np
import salabim as sim


def create_env(capacity=4, num_lifts=3, topfloor=15):
    def do_animation():
        global xvisitor_dim
        global yvisitor_dim
        global xlift
        global capacity_last, num_lifts_last, topfloor_last

        env.modelname("Elevator")
        env.speed(32)
        env.background_color("20%gray")
        if make_video:
            env.video("Elevator.mp4")

        xvisitor_dim = 30
        yvisitor_dim = xvisitor_dim
        yfloor0 = 20

        # x positions dict for all lifts
        xlift = {}

        xled = {}

        x = env.width()
        for lift in lifts:
            x -= (capacity + 1) * xvisitor_dim
            xlift[lift] = x
        x -= xvisitor_dim
        xsign = x
        x -= xvisitor_dim / 2
        for direction in (up, down):
            x -= xvisitor_dim / 2
            xled[direction] = x
        x -= xvisitor_dim
        xwait = x

        for floor in floors:
            y = yfloor0 + floor.n * yvisitor_dim
            floor.y = y
            for direction in (up, down):
                if (direction == up and floor.n < topfloor) or (direction == down and floor.n > 0):
                    b = xvisitor_dim / 4

                    # print triangle for direction in front of each floor
                    animate_led = sim.AnimatePolygon(
                        spec=(-b, -b, b, -b, 0, b),
                        x=xled[direction],
                        y=y + 2 * b,
                        angle=0 if direction == up else 180,
                        fillcolor=direction_color(direction),
                        visible=lambda arg, t: (arg.floor_direction) in requests,
                    )
                    animate_led.floor_direction = (floor, direction)

            # floor separation line
            sim.AnimateLine(x=0, y=y, spec=(0, 0, xwait, 0))

            # print floor number in front of each floor
            sim.AnimateText(x=xsign, y=y + yvisitor_dim / 2, text=str(floor.n), fontsize=xvisitor_dim / 2)

            # print the waiting line in front of each floor
            sim.AnimateQueue(queue=floor.visitors, x=xwait - xvisitor_dim, y=floor.y, direction="w", title="")

        for lift in lifts:
            x = xlift[lift]

            # draw the lift
            lift.pic = sim.AnimateRectangle(
                x=x, y=lift.y, spec=(0, 0, capacity * xvisitor_dim, yvisitor_dim), fillcolor="lightblue", linewidth=0
            )

            # print visitors in this lift
            sim.AnimateQueue(queue=lift.visitors, x=xlift[lift], y=lift.y, direction="e", title=lift.name(), arg=lift)

        num_lifts_last = num_lifts
        sim.AnimateSlider(
            x=510,
            y=0,
            width=90,
            height=20,
            vmin=1,
            vmax=5,
            resolution=1,
            v=num_lifts,
            label="#elevators",
            action=set_num_lifts,
            xy_anchor="nw",
        )

        topfloor_last = topfloor
        sim.AnimateSlider(
            x=610,
            y=0,
            width=90,
            height=20,
            vmin=5,
            vmax=20,
            resolution=1,
            v=topfloor,
            label="top floor",
            action=set_topfloor,
            xy_anchor="nw",
        )

        capacity_last = capacity
        sim.AnimateSlider(
            x=710,
            y=0,
            width=90,
            height=20,
            vmin=2,
            vmax=6,
            resolution=1,
            v=capacity,
            label="capacity",
            action=set_capacity,
            xy_anchor="nw",
        )

        sim.AnimateSlider(
            x=510,
            y=-50,
            width=90,
            height=25,
            vmin=0,
            vmax=400,
            resolution=25,
            v=load_0_n,
            label="Load 0->n",
            action=set_load_0_n,
            xy_anchor="nw",
        )

        sim.AnimateSlider(
            x=610,
            y=-50,
            width=90,
            height=25,
            vmin=0,
            vmax=400,
            resolution=25,
            v=load_n_n,
            label="Load n->n",
            action=set_load_n_n,
            xy_anchor="nw",
        )

        sim.AnimateSlider(
            x=710,
            y=-50,
            width=90,
            height=25,
            vmin=0,
            vmax=400,
            resolution=25,
            v=load_n_0,
            label="Load n->0",
            action=set_load_n_0,
            xy_anchor="nw",
        )

        env.animate(True)

    def set_load_0_n(val):
        global load_0_n
        load_0_n = float(val)
        if vg_0_n.ispassive():
            vg_0_n.activate()

    def set_load_n_n(val):
        global load_n_n
        load_n_n = float(val)
        if vg_n_n.ispassive():
            vg_n_n.activate()

    def set_load_n_0(val):
        global load_n_0
        load_n_0 = float(val)
        if vg_n_0.ispassive():
            vg_n_0.activate()

    def set_capacity(val):
        global capacity
        global capacity_last
        capacity = int(val)
        if capacity != capacity_last:
            capacity_last = capacity
            env.main().activate()

    def set_num_lifts(val):
        global num_lifts
        global num_lifts_last
        num_lifts = int(val)
        if num_lifts != num_lifts_last:
            num_lifts_last = num_lifts
            env.main().activate()

    def set_topfloor(val):
        global topfloor
        global topfloor_last
        topfloor = int(val)
        if topfloor != topfloor_last:
            topfloor_last = topfloor
            env.main().activate()

    def direction_color(direction):
        if direction == 1:
            return "red"
        if direction == -1:
            return "green"
        return "yellow"

    class VisitorGenerator(sim.Component):
        def setup(self, from_, to, id):
            self.from_ = from_
            self.to = to
            self.id = id

        def process(self):
            while True:
                from_ = random.randint(self.from_[0], self.from_[1])

                # while is just to avoid having similar from and to
                while True:
                    to = random.randint(self.to[0], self.to[1])
                    if from_ != to:
                        break

                Visitor(from_=from_, to=to)
                if self.id == "0_n":
                    load = load_0_n
                elif self.id == "n_0":
                    load = load_n_0
                else:
                    load = load_n_n

                if load == 0:
                    yield self.passivate()
                else:
                    iat = 3600 / load
                    r = random.uniform(0.5, 1.5)
                    yield self.hold(r * iat)

    class Visitor(sim.Component):
        def setup(self, from_, to):
            self.fromfloor = floors[from_]
            self.tofloor = floors[to]
            self.direction = getdirection(self.fromfloor, self.tofloor)

        def animation_objects(self, q):
            size_x = xvisitor_dim
            size_y = yvisitor_dim
            b = 0.1 * xvisitor_dim
            an0 = sim.AnimateRectangle(
                spec=(b, 2, xvisitor_dim - b, yvisitor_dim - b),
                linewidth=0,
                fillcolor=direction_color(self.direction),
                text=str(self.tofloor.n),
                fontsize=xvisitor_dim * 0.7,
                textcolor="white",
            )
            return size_x, size_y, an0

        def process(self):
            self.enter(self.fromfloor.visitors)
            if not (self.fromfloor, self.direction) in requests:
                requests[self.fromfloor, self.direction] = self.env.now()
            for car in lifts:
                if car.ispassive():
                    car.activate()

            yield self.passivate()

    class VisitorsInCar(sim.Queue):
        pass

    class Lift(sim.Component):
        def setup(self):
            self.capacity = capacity
            self.direction = still
            self.floor = floors[0]
            self.dooropen = False
            self.visitors = VisitorsInCar()
            self.neural_control = True

        def y(self, t):
            if self.mode() == "Move":
                y = sim.interpolate(t, self.mode_time(), self.scheduled_time(), self.floor.y, self.nextfloor.y)
            else:
                y = self.floor.y
            return y

        def process(self):
            self.floor = floors[0]
            self.direction = still
            self.dooropen = False

            while True:

                if self.direction == still:
                    if not requests:
                        yield self.passivate(mode="Idle")

                # unloading the visitors who reached the desired floor
                if self.count_to_floor(self.floor) > 0:
                    yield self.hold(dooropen_time, mode="Door open")
                    self.dooropen = True
                    for visitor in self.visitors:
                        if visitor.tofloor == self.floor:
                            visitor.leave(self.visitors) # the visitor leaves the lift visitors queue
                            visitor.activate()
                    yield self.hold(exit_time, mode="Let exit")

                if self.direction == still:
                    self.direction = up  # just random

                # loading the lift
                for self.direction in (self.direction, -self.direction):

                    # we delete the call of the lift because it has arrived
                    if (self.floor, self.direction) in requests:
                        del requests[self.floor, self.direction]

                        if not self.dooropen:
                            yield self.hold(dooropen_time, mode="Door open")
                            self.dooropen = True
                        for visitor in self.floor.visitors:
                            # only pickup the visitors going the same way as the lift
                            if visitor.direction == self.direction:
                                # we still have some room in the lift
                                if len(self.visitors) < self.capacity:
                                    # the visitor leaves the floor queue
                                    visitor.leave(self.floor.visitors)
                                    # to join the lift
                                    visitor.enter(self.visitors)
                            yield self.hold(enter_time, mode="Let in")
                        if self.floor.count_in_direction(self.direction) > 0:
                            if not (self.floor, self.direction) in requests:
                                requests[self.floor, self.direction] = self.env.now()

                    if self.neural_control is False:
                        if self.visitors:
                            break
                        else:
                            # we select the earliest request to serve it first by defining self.direction accordingly
                            if requests:
                                earliest = sim.inf
                                for (floor, direction) in requests:
                                    if requests[floor, direction] < earliest:
                                        self.direction = getdirection(self.floor, floor)
                                        earliest = requests[floor, direction]
                            else:
                                # or keep the lift still if no requests
                                self.direction = still

                if self.dooropen:
                    yield self.hold(doorclose_time, mode="Door close")
                    self.dooropen = False

                if self.direction != still:
                    # define the next target floor
                    print(f'go to the {self.floor.n + self.direction}th floor')
                    if len(floors) >= (self.floor.n + self.direction) >= 0:
                        self.nextfloor = floors[self.floor.n + self.direction]
                        yield self.hold(move_time, mode="Move")
                        self.floor = self.nextfloor
                    else:
                        print('out of range')

        def count_to_floor(self, tofloor):
            n = 0
            for visitor in self.visitors:
                if visitor.tofloor == tofloor:
                    n += 1
            return n

    class Visitors(sim.Queue):
        pass

    class Floor:
        def __init__(self):
            self.visitors = Visitors()
            self.n = self.visitors.sequence_number()

        def count_in_direction(self, dir):
            n = 0
            for visitor in self.visitors:
                if visitor.direction == dir:
                    n += 1
            return n

    def getdirection(fromfloor, tofloor):
        if fromfloor.n < tofloor.n:
            return +1
        if fromfloor.n > tofloor.n:
            return -1
        return 0

    up = 1
    still = 0
    down = -1

    move_time = 10
    dooropen_time = 3
    doorclose_time = 3
    enter_time = 3
    exit_time = 3

    load_0_n = 50
    load_n_n = 100
    load_n_0 = 100

    env = sim.Environment()

    vg_0_n = VisitorGenerator(from_=(0, 0), to=(1, topfloor), id="0_n", name="vg_0_n")
    vg_n_0 = VisitorGenerator(from_=(1, topfloor), to=(0, 0), id="n_0", name="vg_n_0")
    vg_n_n = VisitorGenerator(from_=(1, topfloor), to=(1, topfloor), id="n_n", name="vg_n_n")

    requests = {}

    floors = [Floor() for ifloor in range(topfloor+1)]
    env.define_floors(floors)

    lifts = [Lift() for ilift in range(num_lifts)]
    env.define_lifts(lifts)

    make_video = False

    do_animation()
    return env
