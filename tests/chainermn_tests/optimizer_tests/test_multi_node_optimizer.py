import chainer
import chainer.testing
import chainer.testing.attr
import chainermn
import mock
import numpy as np
import unittest


class ExampleModel(chainer.Chain):

    def __init__(self):
        super(ExampleModel, self).__init__(
            a=chainer.links.Linear(2, 3),
            b=chainer.links.Linear(3, 4),
            c=chainer.links.Linear(4, 5),
        )


class TestMultiNodeOptimizer(unittest.TestCase):

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
        self.assertEqual(actual_optimizer.t, 1)
        optimizer.target.a.W.update_rule.update.assert_called_once_with(
            optimizer.target.a.W)
        optimizer.target.b.W.update_rule.update.assert_called_once_with(
            optimizer.target.b.W)
        optimizer.target.c.W.update_rule.update.assert_called_once_with(
            optimizer.target.c.W)

        base = (comm.size - 1.0) / 2
        chainer.testing.assert_allclose(optimizer.target.a.W.grad,
                                        (base + 0) * np.ones((3, 2)))
        chainer.testing.assert_allclose(optimizer.target.b.W.grad,
                                        (base + 1) * np.ones((4, 3)))
        chainer.testing.assert_allclose(optimizer.target.c.W.grad,
                                        (base + 2) * np.ones((5, 4)))

    def test_update_with_cpu(self):
        comm = chainermn.create_communicator('naive')
        model = self.setup_model()
        actual_optimizer = self.setup_actual_optimizer()
        multi_node_optimizer = chainermn.create_multi_node_optimizer(
            actual_optimizer, comm)
        self.check_update(comm, model, actual_optimizer, multi_node_optimizer)

    @chainer.testing.attr.gpu
    def test_update_with_gpu(self):
        comm = chainermn.create_communicator('hierarchical')
        device = self.comm.intra_rank
        chainer.cuda.get_device(device).use()
        model = self.setup_model()
        model.to_gpu()
        actual_optimizer = self.setup_actual_optimizer()
        multi_node_optimizer = chainermn.create_multi_node_optimizer(
            actual_optimizer, comm)
        self.check_update(comm, model, actual_optimizer, multi_node_optimizer)


class DynamicExampleModel(chainer.Chain):

    def __init__(self):
        super(DynamicExampleModel, self).__init__()
        with self.init_scope():
            self.a = chainer.links.Linear(2, 3)
            self.b = chainer.links.Linear(3, 4)


class TestMultiNodeOptimizerWithDynamicModel(unittest.TestCase):

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

    def check_update(self, device, comm, model, actual_optimizer, optimizer):
        optimizer.setup(model)
        optimizer.target.a.W.data[:] = comm.rank
        optimizer.target.b.W.data[:] = comm.rank + 1
        optimizer.update()
        self.assertEqual(actual_optimizer.t, 0)
        chainer.testing.assert_allclose(optimizer.target.a.W.data,
                                        0 * np.ones((3, 2)))
        chainer.testing.assert_allclose(optimizer.target.b.W.data,
                                        1 * np.ones((4, 3)))

        with model.init_scope():
            c = chainer.links.Linear(4, 5)
            if device:
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

        optimizer.target.a.W.grad[:] = comm.rank
        optimizer.target.b.W.grad[:] = comm.rank + 1
        optimizer.target.c.W.grad[:] = comm.rank + 2

        optimizer.update()
        self.assertEqual(actual_optimizer.t, 1)
        optimizer.target.a.W.update_rule.update.assert_called_once_with(
            optimizer.target.a.W)
        optimizer.target.b.W.update_rule.update.assert_called_once_with(
            optimizer.target.b.W)
        optimizer.target.c.W.update_rule.update.assert_called_once_with(
            optimizer.target.c.W)

        base = (comm.size - 1.0) / 2
        chainer.testing.assert_allclose(optimizer.target.a.W.grad,
                                        (base + 0) * np.ones((3, 2)))
        chainer.testing.assert_allclose(optimizer.target.b.W.grad,
                                        (base + 1) * np.ones((4, 3)))
        chainer.testing.assert_allclose(optimizer.target.c.W.grad,
                                        (base + 2) * np.ones((5, 4)))

    def test_update_with_cpu(self):
        comm = chainermn.create_communicator('naive')
        device = -1
        model = self.setup_model()
        actual_optimizer = self.setup_actual_optimizer()
        multi_node_optimizer = chainermn.create_multi_node_optimizer(
            actual_optimizer, comm)
        self.check_update(device, comm, model, actual_optimizer,
                          multi_node_optimizer)

    @chainer.testing.attr.gpu
    def test_update_with_gpu(self):
        comm = chainermn.create_communicator('hierarchical')
        device = comm.intra_rank
        chainer.cuda.get_device(device).use()
        model = self.setup_model()
        model.to_gpu()
        actual_optimizer = self.setup_actual_optimizer()
        multi_node_optimizer = chainermn.create_multi_node_optimizer(
            actual_optimizer, comm)
        self.check_update(device, comm, model, actual_optimizer,
                          multi_node_optimizer)
