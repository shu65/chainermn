import chainer
import chainer.testing
import chainer.testing.attr
import chainermn
from chainermn import nccl
import mock
import numpy as np
import pytest
import unittest
import mpi4py
import os


class ExampleModel(chainer.Chain):

    def __init__(self):
        super(ExampleModel, self).__init__(
            a=chainer.links.Linear(2, 3),
            b=chainer.links.Linear(3, 4),
            c=chainer.links.Linear(4, 5),
        )


class TestDoubleBufferingOptimizer(unittest.TestCase):

    def setup_model(self):
        model = ExampleModel()
        model.a.W.data[:] = 0
        model.b.W.data[:] = 0
        model.c.W.data[:] = 0
        model.a.W.grad[:] = 0
        model.b.W.grad[:] = 0
        model.c.W.grad[:] = 0
        return model

    def setup_actual_optimizer(self):
        actual_optimizer = chainer.GradientMethod()
        actual_optimizer.create_update_rule = mock.MagicMock
        return actual_optimizer

    def check_update(self, comm, model, actual_optimizer, optimizer):
        optimizer.setup(model)
        optimizer.target.a.W.data[:] = comm.rank
        optimizer.target.b.W.data[:] = comm.rank + 1
        optimizer.target.c.W.data[:] = comm.rank + 2
        optimizer.update()
        self.assertEqual(actual_optimizer.t, 0)
        chainer.testing.assert_allclose(optimizer.target.a.W.data,
                                        0 * np.ones((3, 2)))
        chainer.testing.assert_allclose(optimizer.target.b.W.data,
                                        1 * np.ones((4, 3)))
        chainer.testing.assert_allclose(optimizer.target.c.W.data,
                                        2 * np.ones((5, 4)))

        optimizer.target.a.W.grad[:] = comm.rank
        optimizer.target.b.W.grad[:] = comm.rank + 1
        optimizer.target.c.W.grad[:] = comm.rank + 2

        optimizer.update()
        optimizer.wait()
        self.assertEqual(actual_optimizer.t, 0)
        base = (comm.size - 1.0) / 2
        chainer.testing.assert_allclose(
            optimizer.communicated_target.a.W.grad,
            (base + 0) * np.ones((3, 2)))
        chainer.testing.assert_allclose(
            optimizer.communicated_target.b.W.grad,
            (base + 1) * np.ones((4, 3)))
        chainer.testing.assert_allclose(
            optimizer.communicated_target.c.W.grad,
            (base + 2) * np.ones((5, 4)))

        optimizer.target.a.W.grad[:] = comm.rank + 3
        optimizer.target.b.W.grad[:] = comm.rank + 4
        optimizer.target.c.W.grad[:] = comm.rank + 5
        optimizer.update()
        optimizer.wait()
        self.assertEqual(actual_optimizer.t, 1)

        optimizer.target.a.W.update_rule.update.assert_called_once_with(
            optimizer.target.a.W)
        optimizer.target.b.W.update_rule.update.assert_called_once_with(
            optimizer.target.b.W)
        optimizer.target.c.W.update_rule.update.assert_called_once_with(
            optimizer.target.c.W)
        chainer.testing.assert_allclose(
            optimizer.communicated_target.a.W.grad,
            (base + 3) * np.ones((3, 2)))
        chainer.testing.assert_allclose(
            optimizer.communicated_target.b.W.grad,
            (base + 4) * np.ones((4, 3)))
        chainer.testing.assert_allclose(
            optimizer.communicated_target.c.W.grad,
            (base + 5) * np.ones((5, 4)))

    @chainer.testing.attr.gpu
    def test_update(self):
        if nccl.get_version() < 2000:
            pytest.skip('This test requires NCCL version >= 2.0')
        comm = chainermn.create_communicator('pure_nccl')
        device = comm.intra_rank
        chainer.cuda.get_device(device).use()
        model = self.setup_model()
        model.to_gpu()
        actual_optimizer = self.setup_actual_optimizer()
        multi_node_optimizer = chainermn.create_multi_node_optimizer(
            actual_optimizer, comm, double_buffering=True)
        self.check_update(comm, model, actual_optimizer, multi_node_optimizer)


class DynamicExampleModel(chainer.Chain):

    def __init__(self):
        super(DynamicExampleModel, self).__init__()
        with self.init_scope():
            self.a = chainer.links.Linear(2, 3)
            self.b = chainer.links.Linear(3, 4)


class TestDoubleBufferingOptimizerWithDynamicModel(unittest.TestCase):

    def setup_model(self):
        model = DynamicExampleModel()
        model.a.W.data[:] = 0
        model.b.W.data[:] = 0
        model.a.W.grad[:] = 0
        model.b.W.grad[:] = 0
        return model

    def setup_actual_optimizer(self):
        actual_optimizer = chainer.GradientMethod()
        actual_optimizer.create_update_rule = mock.MagicMock
        return actual_optimizer

    def check_update(self, comm, model, actual_optimizer, optimizer):
        optimizer.setup(model)
        optimizer.target.a.W.data[:] = comm.rank
        optimizer.target.b.W.data[:] = comm.rank + 1
        optimizer.update()
        self.assertEqual(actual_optimizer.t, 0)
        chainer.testing.assert_allclose(optimizer.target.a.W.data,
                                        0 * np.ones((3, 2)))
        chainer.testing.assert_allclose(optimizer.target.b.W.data,
                                        1 * np.ones((4, 3)))

        optimizer.target.a.W.grad[:] = comm.rank
        optimizer.target.b.W.grad[:] = comm.rank + 1

        optimizer.update()
        optimizer.wait()
        self.assertEqual(actual_optimizer.t, 0)
        base = (comm.size - 1.0) / 2
        chainer.testing.assert_allclose(
            optimizer.communicated_target.a.W.grad,
            (base + 0) * np.ones((3, 2)))
        chainer.testing.assert_allclose(
            optimizer.communicated_target.b.W.grad,
            (base + 1) * np.ones((4, 3)))

        optimizer.target.a.W.grad[:] = comm.rank + 3
        optimizer.target.b.W.grad[:] = comm.rank + 4
        optimizer.update()
        optimizer.wait()
        self.assertEqual(actual_optimizer.t, 1)
        optimizer.target.a.W.update_rule.update.assert_called_once_with(
            optimizer.target.a.W)
        optimizer.target.b.W.update_rule.update.assert_called_once_with(
            optimizer.target.b.W)
        chainer.testing.assert_allclose(
            optimizer.communicated_target.a.W.grad,
            (base + 3) * np.ones((3, 2)))
        chainer.testing.assert_allclose(
            optimizer.communicated_target.b.W.grad,
            (base + 4) * np.ones((4, 3)))

        with model.init_scope():
            c = chainer.links.Linear(4, 5)
            c.to_gpu()
            model.c = c
        if comm.rank == 0:
            model.c.W.data[:] = comm.rank + 2
        else:
            model.c.W.data[:] = 0
        optimizer.setup(model)
        optimizer.update()
        self.assertEqual(actual_optimizer.t, 0)
        chainer.testing.assert_allclose(optimizer.target.a.W.data,
                                        0 * np.ones((3, 2)))
        chainer.testing.assert_allclose(optimizer.target.b.W.data,
                                        1 * np.ones((4, 3)))
        chainer.testing.assert_allclose(optimizer.target.c.W.data,
                                        2 * np.ones((5, 4)))

        optimizer.target.a.W.grad[:] = comm.rank + 6
        optimizer.target.b.W.grad[:] = comm.rank + 7
        optimizer.target.c.W.grad[:] = comm.rank + 8

        optimizer.update()
        optimizer.wait()
        self.assertEqual(actual_optimizer.t, 0)
        base = (comm.size - 1.0) / 2
        chainer.testing.assert_allclose(
            optimizer.communicated_target.a.W.grad,
            (base + 6) * np.ones((3, 2)))
        chainer.testing.assert_allclose(
            optimizer.communicated_target.b.W.grad,
            (base + 7) * np.ones((4, 3)))
        chainer.testing.assert_allclose(
            optimizer.communicated_target.c.W.grad,
            (base + 8) * np.ones((5, 4)))

        optimizer.target.a.W.grad[:] = comm.rank + 9
        optimizer.target.b.W.grad[:] = comm.rank + 10
        optimizer.target.c.W.grad[:] = comm.rank + 11
        optimizer.update()
        optimizer.wait()
        self.assertEqual(actual_optimizer.t, 1)
        optimizer.target.a.W.update_rule.update.assert_called_once_with(
            optimizer.target.a.W)
        optimizer.target.b.W.update_rule.update.assert_called_once_with(
            optimizer.target.b.W)
        optimizer.target.c.W.update_rule.update.assert_called_once_with(
            optimizer.target.c.W)
        chainer.testing.assert_allclose(
            optimizer.communicated_target.a.W.grad,
            (base + 9) * np.ones((3, 2)))
        chainer.testing.assert_allclose(
            optimizer.communicated_target.b.W.grad,
            (base + 10) * np.ones((4, 3)))
        chainer.testing.assert_allclose(
            optimizer.communicated_target.c.W.grad,
            (base + 11) * np.ones((5, 4)))

    @chainer.testing.attr.gpu
    def test_update(self):
        mpi_comm = mpi4py.MPI.COMM_WORLD
        mpi_comm.Barrier()
        if nccl.get_version() < 2000:
            pytest.skip('This test requires NCCL version >= 2.0')
        comm = chainermn.create_communicator('pure_nccl')
        st = os.statvfs("/dev/shm")
        print(comm.rank, st)
        device = comm.intra_rank
        chainer.cuda.get_device(device).use()
        model = self.setup_model()
        model.to_gpu()
        actual_optimizer = self.setup_actual_optimizer()
        multi_node_optimizer = chainermn.create_multi_node_optimizer(
            actual_optimizer, comm, double_buffering=True)
        self.check_update(comm, model, actual_optimizer, multi_node_optimizer)
