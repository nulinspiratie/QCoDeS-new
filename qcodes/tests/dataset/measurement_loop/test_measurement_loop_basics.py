from time import perf_counter, sleep
import numpy as np
import pytest

from qcodes.dataset import MeasurementLoop, Sweep
from qcodes.parameters import Parameter, ManualParameter


@pytest.mark.usefixtures("empty_temp_db", "experiment")
def test_original_dond():
    from qcodes.utils.dataset.doNd import LinSweep, dond

    p1_get = ManualParameter("p1_get", initial_value=1)
    p2_get = ManualParameter("p2_get", initial_value=1)
    p1_set = ManualParameter("p1_set", initial_value=1)
    dond(p1_set, 0, 1, 101, p1_get, p2_get)


def test_create_measurement():
    MeasurementLoop("test")


def test_basic_1d_measurement():
    # Initialize parameters
    p1_get = ManualParameter("p1_get")
    p1_set = ManualParameter("p1_set")

    with MeasurementLoop("test") as msmt:
        for val in Sweep(p1_set, 0, 1, 11):
            assert p1_set() == val
            p1_get(val + 1)
            msmt.measure(p1_get)

    data = msmt.dataset
    assert data.name == "test"
    assert data.parameters == "p1_set,p1_get"

    arrays = data.get_parameter_data()
    data_arrays = arrays["p1_get"]

    assert np.allclose(data_arrays["p1_get"], np.linspace(1, 2, 11))
    assert np.allclose(data_arrays["p1_set"], np.linspace(0, 1, 11))


def test_basic_2d_measurement():
    # Initialize parameters
    p1_get = ManualParameter("p1_get")
    p1_set = ManualParameter("p1_set")
    p2_set = ManualParameter("p2_set")

    with MeasurementLoop("test") as msmt:
        for val in Sweep(p1_set, 0, 1, 11):
            assert p1_set() == val
            for val2 in Sweep(p2_set, 0, 1, 11):
                assert p2_set() == val2
                p1_get(val + 1)
                msmt.measure(p1_get)

    data = msmt.dataset
    assert data.name == "test"
    assert data.parameters == "p1_set,p2_set,p1_get"

    arrays = data.get_parameter_data()
    data_array = arrays["p1_get"]["p1_get"]

    assert np.allclose(data_array, np.tile(np.linspace(1, 2, 11), (11, 1)).transpose())

    assert np.allclose(
        arrays["p1_get"]["p1_set"], np.tile(np.linspace(0, 1, 11), (11, 1)).transpose()
    )

    assert np.allclose(
        arrays["p1_get"]["p2_set"], np.tile(np.linspace(0, 1, 11), (11, 1))
    )


def test_1d_measurement_duplicate_get():
    # Initialize parameters
    p1_get = ManualParameter("p1_get")
    p1_set = ManualParameter("p1_set")

    with MeasurementLoop("test") as msmt:
        for val in Sweep(p1_set, 0, 1, 11):
            assert p1_set() == val
            p1_get(val + 1)
            msmt.measure(p1_get)
            p1_get(val + 0.5)
            msmt.measure(p1_get)

    data = msmt.dataset
    assert data.name == "test"
    assert data.parameters == "p1_set,p1_get,p1_get_1"

    arrays = data.get_parameter_data()

    offsets = {"p1_get": 1, "p1_get_1": 0.5}
    for key in ["p1_get", "p1_get_1"]:
        data_arrays = arrays[key]

        assert np.allclose(data_arrays[key], np.linspace(0, 1, 11) + offsets[key])
        assert np.allclose(data_arrays["p1_set"], np.linspace(0, 1, 11))


def test_1d_measurement_duplicate_getset():
    # Initialize parameters
    p1_get = ManualParameter("p1_get")
    p1_set = ManualParameter("p1_set")

    with MeasurementLoop("test") as msmt:
        for val in Sweep(p1_set, 0, 1, 11):
            assert p1_set() == val
            p1_get(val + 1)
            msmt.measure(p1_get)
        for val in Sweep(p1_set, 0, 1, 11):
            assert p1_set() == val
            p1_get(val + 0.5)
            msmt.measure(p1_get)

    data = msmt.dataset
    assert data.name == "test"
    assert data.parameters == "p1_set,p1_get,p1_set_1,p1_get_1"

    arrays = data.get_parameter_data()

    offsets = {"p1_get": 1, "p1_get_1": 0.5}
    for suffix in ["", "_1"]:
        get_key = f"p1_get{suffix}"
        set_key = f"p1_set{suffix}"
        data_arrays = arrays[get_key]

        assert np.allclose(
            data_arrays[get_key], np.linspace(0, 1, 11) + offsets[get_key]
        )
        assert np.allclose(data_arrays[set_key], np.linspace(0, 1, 11))


def test_2d_measurement_initialization():
    # Initialize parameters
    p1_get = ManualParameter("p1_get")
    p1_set = ManualParameter("p1_set")
    p2_set = ManualParameter("p2_set")

    with MeasurementLoop("test") as msmt:
        outer_sweep = Sweep(p1_set, 0, 1, 11)
        for k, val in enumerate(outer_sweep):
            assert p1_set() == val

            for val2 in Sweep(p2_set, 0, 1, 11):
                assert p2_set() == val2
                p1_get(val + 1)
                msmt.measure(p1_get)


def test_initialize_empty_dataset():
    from qcodes import Measurement

    msmt = Measurement()
    #   msmt.register_parameter(p1_set)
    #   msmt.register_parameter(p1_get, setpoints=(p1_set,))
    with msmt.run(allow_empty_dataset=True) as datasaver:
        pass


def test_nested_measurement():
    def nested_measurement():
        # Initialize parameters
        p1_get = ManualParameter("p1_get")
        p1_set = ManualParameter("p1_set")

        with MeasurementLoop("test") as msmt:
            for val in Sweep(p1_set, 0, 1, 11):
                assert p1_set() == val
                p1_get(val + 1)
                msmt.measure(p1_get)

    # Initialize parameters
    p2_set = ManualParameter("p2_set")

    with MeasurementLoop("test") as msmt:
        for val2 in Sweep(p2_set, 0, 1, 11):
            assert p2_set() == val2
            nested_measurement()

    data = msmt.dataset
    assert data.name == "test"
    assert data.parameters == "p2_set,p1_set,p1_get"

    arrays = data.get_parameter_data()
    data_array = arrays["p1_get"]["p1_get"]

    assert np.allclose(data_array, np.tile(np.linspace(1, 2, 11), (11, 1)))

    assert np.allclose(
        arrays["p1_get"]["p2_set"], np.tile(np.linspace(0, 1, 11), (11, 1)).transpose()
    )

    assert np.allclose(
        arrays["p1_get"]["p1_set"], np.tile(np.linspace(0, 1, 11), (11, 1))
    )


def test_measurement_no_parameter():
    with MeasurementLoop("test") as msmt:
        for val in Sweep(np.linspace(0, 1, 11), "p1_set", label="p1 label", unit="V"):
            msmt.measure(val + 1, name="p1_get")

    data = msmt.dataset
    assert data.name == "test"
    assert data.parameters == "p1_set,p1_get"

    arrays = data.get_parameter_data()
    data_arrays = arrays["p1_get"]

    assert np.allclose(data_arrays["p1_get"], np.linspace(1, 2, 11))
    assert np.allclose(data_arrays["p1_set"], np.linspace(0, 1, 11))


def test_fraction_complete():
    with MeasurementLoop("test") as msmt:
        print(f'Before Sweep')
        print(f'{msmt.action_indices=}, {msmt.loop_indices=}, {msmt.loop_shape=}')
        for k, val in enumerate(Sweep(np.linspace(0, 1, 10), "p1_set")):
            print(f'\n')
            print(f'\nBefore first measurement')
            print(f'{msmt.action_indices=}, {msmt.loop_indices=}, {msmt.loop_shape=}')
            print(f'{msmt.fraction_complete(silent=False)=}')
            assert msmt.fraction_complete() == round(0.1*k, 3)

            msmt.measure(val+1, name="p1_get")
            print(f'\nBetween first and second measurement')
            print(f'{msmt.action_indices=}, {msmt.loop_indices=}, {msmt.loop_shape=}')
            print(f'{msmt.fraction_complete(silent=False)=}')
            if not k:
                assert msmt.fraction_complete() == 0.1
            else:
                assert msmt.fraction_complete() == round(0.1*k+0.05, 3)

            msmt.measure(val+1, name="p1_get")
            print(f'\nAfter second measurement')
            print(f'{msmt.action_indices=}, {msmt.loop_indices=}, {msmt.loop_shape=}')
            # print(f'{msmt.fraction_complete(silent=False)=}')
            print(f'{msmt.fraction_complete(silent=False)=}')
            assert msmt.fraction_complete() == round(0.1 * (k+1), 3)

        for k, val in enumerate(Sweep(np.linspace(0, 1, 10), "p1_set")):
            print(f'\n')
            print(f'\nBefore first measurement')
            print(f'{msmt.action_indices=}, {msmt.loop_indices=}, {msmt.loop_shape=}')
            print(f'{msmt.fraction_complete(silent=-1)=}')
            assert msmt.fraction_complete() == round(0.5 + 0.05*k, 3)

            msmt.measure(val+1, name="p1_get")
            print(f'\nBetween first and second measurement')
            print(f'{msmt.action_indices=}, {msmt.loop_indices=}, {msmt.loop_shape=}')
            print(f'{msmt.fraction_complete(silent=-1)=}')
            if not k:
                assert msmt.fraction_complete() == 0.55
            else:
                assert msmt.fraction_complete() == round(0.525 + 0.05*k, 3)

            msmt.measure(val+1, name="p1_get")
            print(f'\nAfter second measurement')
            print(f'{msmt.action_indices=}, {msmt.loop_indices=}, {msmt.loop_shape=}')
            # print(f'{msmt.fraction_complete(silent=False)=}')
            print(f'{msmt.fraction_complete(silent=-1)=}')
            assert msmt.fraction_complete() == round(0.5 + 0.05 * (k+1), 3)


@pytest.mark.usefixtures("empty_temp_db", "experiment")
def test_measurement_threaded_sweep():
    def delayed_get(delay=0.1, return_val=42):
        sleep(delay)
        return return_val

    delayed_param = Parameter('delayed_param', get_cmd=delayed_get)

    with MeasurementLoop("test_nonthreaded") as msmt:
        t0 = perf_counter()
        for k in Sweep(range(10), 'sweep'):
            msmt.measure(delayed_param)
        t1 = perf_counter()
    # Ten measurements with 0.1s delay each
    assert t1 - t0 > 0.9

    with MeasurementLoop("test_threaded") as msmt:
        t0 = perf_counter()
        for k in Sweep(range(10), 'sweep'):
            msmt.measure(delayed_param, thread=True)
        t1 = perf_counter()
    # Ten measurements with 0.1s delay each
    assert t1 - t0 < 0.1

    msmt.dataset

