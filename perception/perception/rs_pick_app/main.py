from camera_loop import CameraLoop
from camera_events import CameraEvents
from pick_pipeline import PickPipeline
from coordinate import Coordinate


def run_app():
    coord = Coordinate()
    pipeline = PickPipeline(coord=coord)

    events = CameraEvents()
    loop = CameraLoop(events=events, pipeline=pipeline)

    loop.run()


if __name__ == "__main__":
    run_app()
