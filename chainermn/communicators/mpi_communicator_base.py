import collections

import mpi4py
import numpy

import chainer.cuda
import chainer.utils
from chainermn.communicators import _communication_utility
from chainermn.communicators._communication_utility import chunked_bcast_obj
from chainermn.communicators import _memory_utility
from chainermn.communicators import communicator_base


def _cnt_to_dsp(cnt):
    """Utility to convert length array to cumulative array."""
    return [0] + numpy.cumsum(cnt)[:-1].tolist()


class _MessageType(object):

    def __init__(self, obj):
        if isinstance(obj, numpy.ndarray) \
                or chainer.cuda.get_array_module(obj) is not numpy:
            self.is_tuple = False
            self.narr = 1
            self.ndims = [obj.ndim]
            self.shapes = [obj.shape]
        elif isinstance(obj, collections.Iterable):
            self.is_tuple = True
            self.narr = len(obj)
            self.ndims = [x.ndim for x in obj]
            self.shapes = [x.shape for x in obj]
        else:
            raise ValueError(
                'Message object must be numpy/cupy array or tuple.')


class MpiCommunicatorBase(communicator_base.CommunicatorBase):
    '''MpiCommunicatorBase

    Implementation of communicator interface defined by
    :class:`CommunicatorBase`. This communicator assumes MPI4py and
    all ChainerMN processes are invoked by ``mpirun`` (``mpiexec``)
    command. Although this lacks several important methods such as
    ``allreduce_grad`` to be impelmented with speficic algorithm. See
    hierarcical communicator or pure_nccl communicator for example.

    '''

    def __init__(self, mpi_comm):
        self.mpi_comm = mpi_comm
        self._init_ranks()

    @property
    def rank(self):
        return self.mpi_comm.rank

    @property
    def intra_rank(self):
        return self._intra_rank

    @property
    def size(self):
        return self.mpi_comm.size

    def split(self, color, key):
        return self.__class__(mpi_comm=self.mpi_comm.Split(color, key))

    def alltoall(self, xs):
        """A primitive of inter-process all-to-all function.

        This method tries to invoke all-to-all communication within the
        communicator. All processes in the communicator are expected to
        invoke ``alltoall()``. This method relies on mpi4py fast communication
        optimized for numpy arrays, as well as ``send()`` and ``recv()``.

        Args:
            xs (tuple of numpy.ndarray)

        Returns:
            ys (tuple of numpy.ndarray):
                Received arrays. The length of tuple equals to
                the communicator size.
        """
        chainer.utils.experimental(
            'chainermn.communicators.MpiCommunicatorBase.alltoall')

        if len(xs) != self.size:
            raise ValueError(
                'The length of data must be same as communicator size.')

        # Type check.
        for x in xs:
            if x.dtype != numpy.float32:
                raise ValueError(
                    'alltoall only support dtype == numpy.float32')

        # Mediate #axes of arrays.
        sndims = numpy.array([x.ndim for x in xs], dtype=numpy.int32)
        rndims = numpy.empty(self.size, dtype=numpy.int32)
        self.mpi_comm.Alltoall(
            [sndims, mpi4py.MPI.INT],
            [rndims, mpi4py.MPI.INT])

        # Arbitrate shapes of arrays.
        sshapes = numpy.hstack([x.shape for x in xs]).astype(numpy.int32)
        rshapes = numpy.empty(sum(rndims), dtype=numpy.int32)
        self.mpi_comm.Alltoallv(
            [sshapes, (sndims, _cnt_to_dsp(sndims)), mpi4py.MPI.INT],
            [rshapes, (rndims, _cnt_to_dsp(rndims)), mpi4py.MPI.INT])
        shapes = [rshapes[i:i + l]
                  for i, l in zip(_cnt_to_dsp(rndims), rndims)]

        # Collective communication.
        slens = [numpy.prod(x.shape) for x in xs]
        xp = chainer.cuda.get_array_module(xs[0])
        sbuf = xp.hstack([x.reshape(-1) for x in xs])
        rlens = [numpy.prod(s) for s in shapes]
        rbuf = numpy.empty(sum(rlens), dtype=numpy.float32)
        if xp is not numpy:
            sbuf = _memory_utility.array_to_buffer_object(sbuf)[0]
            chainer.cuda.Stream.null.synchronize()
        self.mpi_comm.Alltoallv(
            [sbuf, (slens, _cnt_to_dsp(slens)), mpi4py.MPI.FLOAT],
            [rbuf, (rlens, _cnt_to_dsp(rlens)), mpi4py.MPI.FLOAT])
        ys = [rbuf[i:i + l].reshape(s)
              for i, l, s in zip(_cnt_to_dsp(rlens), rlens, shapes)]

        return tuple(ys)

    def send(self, data, dest, tag):
        """A primitive for inter-process transmitter.

        This method sends numpy-array to target process.
        The target process is expected to invoke ``recv()``.
        This method relies on mpi4py fast communication optimized for
        numpy arrays, which discards any information attached to
        chainer.Variable objects. Please be sure.

        Args:
            data: data to be sent (tuple, list or raw numpy/cupy array)
            dest (int): Target process specifier.
            tag (int): Message ID (MPI feature).

        """
        chainer.utils.experimental(
            'chainermn.communicators.MpiCommunicatorBase.send')

        msgtype = _MessageType(data)
        """We use ssend() instead of send() to pass unittests.
        If we don't use it, an error occurs in
        test_point_to_point_communication.py
        when using MVAPICH2-2.2 and GPUs.
        """
        self.mpi_comm.ssend(msgtype, dest=dest, tag=tag)

        # Type check.
        if not msgtype.is_tuple:
            data = [data]

        for x in data:
            if x.dtype != numpy.float32:
                raise ValueError('send only support dtype == numpy.float32')

        for array in data:
            if chainer.cuda.get_array_module(array) is not numpy:
                chainer.cuda.Stream.null.synchronize()

            buf = _memory_utility.array_to_buffer_object(array)
            """We use Ssend() for the same reason as using ssend()."""
            self.mpi_comm.Ssend(buf, dest=dest, tag=tag)

    def recv(self, source, tag):
        """A primitive of inter-process receiver.

        This method tries to receive numpy-array from target process.
        The target process is expected to invoke ``send()``.
        This method relies on mpi4py fast communication optimized for
        numpy arrays, which discards any information attached to
        chainer.Variable objects. Please be sure.

        Args:
            source (int): Target process specifier.
            tag (int): Message ID (MPI feature).

        """

        chainer.utils.experimental(
            'chainermn.communicators.MpiCommunicatorBase.recv')

        msgtype = self.mpi_comm.recv(source=source, tag=tag)

        if msgtype.is_tuple:
            msg = []
            for shape in msgtype.shapes:
                buf = numpy.empty(numpy.prod(shape), dtype=numpy.float32)
                self.mpi_comm.Recv(buf, source=source, tag=tag)
                msg.append(buf.reshape(shape))
            return tuple(msg)

        else:
            assert len(msgtype.shapes) == 1
            shape = msgtype.shapes[0]
            buf = numpy.empty(numpy.prod(shape), dtype=numpy.float32)
            self.mpi_comm.Recv(buf, source=source, tag=tag)
            return buf.reshape(shape)

    def bcast(self, x, root=0):
        """A primitive of inter-process broadcast communication.

        This method tries to invoke broadcast communication within the
        communicator. All processes in the communicator are expected to
        invoke ``broadcast()``. This method relies on mpi4py fast communication
        optimized for numpy arrays, as well as ``send()`` and ``recv()``.

        Args:
            x (numpy.array): Array to be broadcasted.

        Returns:
            ys (tuple of numpy.ndarray): Received arrays.
        """
        chainer.utils.experimental(
            'chainermn.communicators.MpiCommunicatorBase.bcast')

        is_master = self.mpi_comm.rank == root

        if is_master:
            msgtype = _MessageType(x)
            if msgtype.is_tuple:
                raise TypeError('Tuple data cannot be broadcasted')

            elif x.dtype != numpy.float32:
                raise ValueError(
                    'MPI broadcast only supports dtype == numpy.float32')

            msgtype = self.mpi_comm.bcast(msgtype, root)
            shape = msgtype.shapes[0]
            buf = _memory_utility.array_to_buffer_object(x)
            self.mpi_comm.Bcast(buf, root)
            return x
        else:
            msgtype = None
            msgtype = self.mpi_comm.bcast(msgtype, root)
            shape = msgtype.shapes[0]
            buf = numpy.empty(numpy.prod(shape), dtype=numpy.float32)
            self.mpi_comm.Bcast(buf, root)
            return buf.reshape(shape)

    def gather(self, x, root=0):
        """A primitive of inter-process gather communication.

        This method tries to invoke gather communication within the
        communicator. All processes in the communicator are expected to
        invoke ``gather()``. This method relies on mpi4py fast communication
        optimized for numpy arrays, as well as ``send()`` and ``recv()``.

        Note that this method can only handle the same shapes of data
        over all processes, and cannot handle tuple data.

        Args:
            x (numpy.array): Array to be gathered.

        Returns:
            ys (numpy.ndarray):
                Received arrays with shape (#proc, [data-shape]).
                ``None`` for non-root processes.
        """
        chainer.utils.experimental(
            'chainermn.communicators.MpiCommunicatorBase.gather')

        is_master = self.mpi_comm.rank == root

        msgtype = _MessageType(x)
        msgtypes = self.mpi_comm.gather(msgtype, root)

        # Type check.
        if is_master:
            shape = msgtype.shapes[0]
            for msgtype in msgtypes:
                if msgtype.is_tuple:
                    raise TypeError('gather cannot handle tuple data')

                assert len(msgtype.shapes) == 1

                if msgtype.shapes[0] != shape:
                    raise ValueError(
                        'gather cannot handle different shapes of data')

        if x.dtype != numpy.float32:
            raise ValueError('gather only support dtype == numpy.float32')

        # Gather data.
        sbuf = _memory_utility.array_to_buffer_object(x)
        if is_master:
            shape = tuple([self.mpi_comm.size]) + shape
            rbuf = numpy.empty(numpy.prod(shape), dtype=numpy.float32)
        else:
            rbuf = None
        self.mpi_comm.Gather(sbuf, rbuf, root)

        if is_master:
            return rbuf.reshape(shape)
        else:
            return None

    # Objects
    def send_obj(self, obj, dest):
        self.mpi_comm.send(obj, dest=dest)

    def recv_obj(self, source):
        return self.mpi_comm.recv(source=source)

    def bcast_obj(self, obj, max_buf_len=256 * 1024 * 1024, root=0):
        return chunked_bcast_obj(obj, self.mpi_comm,
                                 max_buf_len=max_buf_len,
                                 root=root)

    def gather_obj(self, obj, root=0):
        return self.mpi_comm.gather(obj, root=root)

    def allreduce_obj(self, obj):
        # Summation by default
        return self.mpi_comm.allreduce(obj)

    def bcast_data(self, model):
        for _, param in sorted(model.namedparams()):
            buf = _memory_utility.array_to_buffer_object(param.data)
            self.mpi_comm.Bcast(buf)

    # Private methods
    def _init_ranks(self):
        my_ranks = _communication_utility.init_ranks(self.mpi_comm)
        assert my_ranks[0] == self.mpi_comm.rank
        self._intra_rank = my_ranks[1]
        self.intra_size = my_ranks[2]
        self.inter_rank = my_ranks[3]
        self.inter_size = my_ranks[4]
