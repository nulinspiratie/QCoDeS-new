import contextlib
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from qcodes import ManualParameter, Parameter
from qcodes.dataset import initialise_or_create_database_at, load_or_create_experiment
from qcodes.dataset.data_set import load_by_id
from qcodes.dataset.measurement_loop import MeasurementLoop, Sweep
from qcodes.utils.dataset.doNd import LinSweep

# def get_data_array(dataset, label):


@pytest.fixture
def create_dummy_database():
    @contextlib.contextmanager
    def func_context_manager():
        with tempfile.TemporaryDirectory() as temporary_folder:
            temporary_folder = tempfile.TemporaryDirectory()
            print(f"Created temporary folder for database: {temporary_folder}")

            assert Path(temporary_folder.name).exists()
            db_path = Path(temporary_folder.name) / "test_database.db"
            initialise_or_create_database_at(str(db_path))

            yield load_or_create_experiment("test_experiment")

    return func_context_manager


def test_create_measurement(create_dummy_database):
    with create_dummy_database():
        MeasurementLoop("test")


def test_basic_1D_measurement(create_dummy_database):
    with create_dummy_database():
        # Initialize parameters
        p1_get = ManualParameter("p1_get")
        p1_set = ManualParameter("p1_set")

        with MeasurementLoop("test") as msmt:
            for val in Sweep(LinSweep(p1_set, 0, 1, 11)):
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


def test_basic_2D_measurement(create_dummy_database):
    with create_dummy_database():
        # Initialize parameters
        p1_get = ManualParameter("p1_get")
        p1_set = ManualParameter("p1_set")
        p2_set = ManualParameter("p2_set")

        with MeasurementLoop("test") as msmt:
            for val in Sweep(LinSweep(p1_set, 0, 1, 11)):
                assert p1_set() == val
                for val2 in Sweep(LinSweep(p2_set, 0, 1, 11)):
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


def test_1D_measurement_duplicate_get(create_dummy_database):
    with create_dummy_database():
        # Initialize parameters
        p1_get = ManualParameter("p1_get")
        p1_set = ManualParameter("p1_set")

        with MeasurementLoop("test") as msmt:
            for val in Sweep(LinSweep(p1_set, 0, 1, 11)):
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


def test_1D_measurement_duplicate_getset(create_dummy_database):
    with create_dummy_database():
        # Initialize parameters
        p1_get = ManualParameter("p1_get")
        p1_set = ManualParameter("p1_set")

        with MeasurementLoop("test") as msmt:
            for val in Sweep(LinSweep(p1_set, 0, 1, 11)):
                assert p1_set() == val
                p1_get(val + 1)
                msmt.measure(p1_get)
            for val in Sweep(LinSweep(p1_set, 0, 1, 11)):
                assert p1_set() == val
                p1_get(val + 0.5)
                msmt.measure(p1_get)

    data = msmt.dataset
    assert data.name == "test"
    assert data.parameters == "p1_set,p1_get,p1_set_1,p1_get_1"

    arrays = data.get_parameter_data()

    offsets = {"p1_get": 1, "p1_get_1": 0.5}
    for suffix in ['', '_1']:
        get_key = f"p1_get{suffix}"
        set_key = f"p1_set{suffix}"
        data_arrays = arrays[get_key]

        assert np.allclose(
            data_arrays[get_key], np.linspace(0, 1, 11) + offsets[get_key]
        )
        assert np.allclose(data_arrays[set_key], np.linspace(0, 1, 11))


def test_2D_measurement_initialization(create_dummy_database):
    with create_dummy_database():
        # Initialize parameters
        p1_get = ManualParameter("p1_get")
        p1_set = ManualParameter("p1_set")
        p2_set = ManualParameter("p2_set")

        with MeasurementLoop("test") as msmt:
            outer_sweep = Sweep(LinSweep(p1_set, 0, 1, 11))
            for k, val in enumerate(outer_sweep):
                assert p1_set() == val

                for val2 in Sweep(LinSweep(p2_set, 0, 1, 11)):
                    assert p2_set() == val2
                    p1_get(val + 1)
                    msmt.measure(p1_get)


def test_dataset_registering_shared_set(create_dummy_database):
    from qcodes import Measurement

    with create_dummy_database():
        p1_get = ManualParameter("p1_get")
        p1_set = ManualParameter("p1_set")

        p2_get = ManualParameter("p2_get")

        msmt = Measurement()
        msmt.register_parameter(p1_set)
        msmt.register_parameter(p1_get, setpoints=(p1_set,))
        # TODO allow cache
        with msmt.run(in_memory_cache=False) as datasaver:
            dataset = datasaver.dataset

            for k, set_v in enumerate(np.linspace(0, 25, 10)):
                p1_set(set_v)
                datasaver.add_result((p1_set, set_v), (p1_get, 123))
                if not k:
                    msmt.register_parameter(p2_get, setpoints=(p1_set,))
                  #   update_interdependencies(msmt, datasaver)

                datasaver.add_result((p1_set, set_v), (p2_get, 124))

        loaded_dataset = load_by_id(dataset.run_id)
        run_description = loaded_dataset._get_run_description_from_db()
        print(run_description)


def test_dataset_registering_separate_set(create_dummy_database):
    from qcodes import Measurement

    with create_dummy_database():
        p1_get = ManualParameter("p1_get")
        p1_set = ManualParameter("p1_set")

        p2_set = ManualParameter("p2_set")
        p2_get = ManualParameter("p2_get")

        msmt = Measurement()
        msmt.register_parameter(p1_set)
        msmt.register_parameter(p1_get, setpoints=(p1_set,))
        # TODO allow cache
        with msmt.run(in_memory_cache=False) as datasaver:
            dataset = datasaver.dataset

            for set_v in np.linspace(0, 25, 10):
                p1_set(set_v)
                datasaver.add_result((p1_set, set_v), (p1_get, 123))

            # Add new parameters
            msmt.register_parameter(p2_set)
            msmt.register_parameter(p2_get, setpoints=(p2_set,))
            # update_interdependencies(msmt, datasaver)

            print(msmt._interdeps)
            for set_v in np.linspace(0, 25, 10):
                p2_set(set_v)
                datasaver.add_result((p2_set, set_v), (p2_get, 124))

        loaded_dataset = load_by_id(dataset.run_id)
        run_description = loaded_dataset._get_run_description_from_db()
        print(run_description)


def test_initialize_empty_dataset(create_dummy_database):
    from qcodes import Measurement

    with create_dummy_database():
        msmt = Measurement()
      #   msmt.register_parameter(p1_set)
      #   msmt.register_parameter(p1_get, setpoints=(p1_set,))
        with msmt.run(allow_empty_dataset=True) as datasaver:
           pass
