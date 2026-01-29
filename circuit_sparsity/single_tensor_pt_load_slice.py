import io
import struct
import pickle
import torch
import numpy as np


def read_tensor_slice_from_file(f, func, readslice):
    # TODO: support arbitrary slicing. rn this only supports 
    # reading something like tensor[a, b, c, :, :] (given readslice (a, b, c))
    # but this isn't a fundamental limitation

    def read_zip_descriptor(chunk):
        descriptorsig = (0x08074b50).to_bytes(4, 'little')
        sig = chunk.read(4)
        assert sig == descriptorsig, f"Not a valid zip file {sig}"
        chunk.read(4 + 4 + 4)  # crc32, compressed size, uncompressed size

    def read_zip_header(chunk):
        zipsig = (0x04034b50).to_bytes(4, 'little')
        sig = chunk.read(4)
        assert sig == zipsig, f"Not a valid zip file {sig}"
        chunk.read(2 + 2)  # version needed to extract, general purpose bit flag
        compression_method = chunk.read(2)  # compression method
        assert compression_method == b'\x00\x00', "expected uncompressed!"
        chunk.read(2 + 2 + 4)  # last mod file time, last mod file date, crc32
        compressed_size = struct.unpack('<I', chunk.read(4))[0]
        uncompressed_size = struct.unpack('<I', chunk.read(4))[0]
        assert compressed_size == uncompressed_size, "expected uncompressed size to match compressed size"
        filename_length = struct.unpack('<H', chunk.read(2))[0]
        extra_field_length = struct.unpack('<H', chunk.read(2))[0]
        filename = chunk.read(filename_length)
        extra_field = chunk.read(extra_field_length)
        data = chunk.read(uncompressed_size)
        return filename.decode('utf-8'), data

    def read_central_dir_entry(f):
        central_sig = (0x02014b50).to_bytes(4, 'little')
        sig = f.read(4)
        if sig == (0x06064b50).to_bytes(4, 'little'):
            return None, None, None  # reached end of central directory

        assert sig == central_sig, f"Not a valid zip file {sig} at {f.tell():x}"

        f.read(24 - 4)
        file_uncompressed_size = struct.unpack('<I', f.read(4))[0]
        filename_length = struct.unpack('<H', f.read(2))[0]
        extra_field_length = struct.unpack('<H', f.read(2))[0]
        comment_length = struct.unpack('<H', f.read(2))[0]
        f.read(8)  # disk number start, internal file attributes, external file attributes
        offset = struct.unpack('<I', f.read(4))[0]  # relative offset of local header
        filename = f.read(filename_length).decode('utf-8')
        extra_field = f.read(extra_field_length)
        comment = f.read(comment_length).decode('utf-8')

        # zip64
        is_zip64 = (file_uncompressed_size == 0xFFFFFFFF)

        if is_zip64:
            # read the zip64 extra field
            zip64_extra_length = struct.unpack('<H', extra_field[2:4])[0]
            zip64_extra = extra_field[4:4 + zip64_extra_length]
            file_uncompressed_size = struct.unpack('<Q', zip64_extra[0:8])[0]
            offset = struct.unpack('<Q', zip64_extra[8:16])[0]
            return filename, offset, file_uncompressed_size
        else:
            return filename, offset, file_uncompressed_size

    def read_central_directory(f):
        # read the central directory
        f.seek(-512, io.SEEK_END)
        tail_offset = f.tell()
        tail = f.read(512)
        # find EOCD
        eocd_sig = (0x06054b50).to_bytes(4, 'little')
        eocd_zip64_sig = (0x06064b50).to_bytes(4, 'little')
        eocd_pos = tail.rfind(eocd_sig)
        eocd_zip64_pos = tail.rfind(eocd_zip64_sig)
        if eocd_zip64_pos != -1:
            eocd_pos = eocd_zip64_pos
            is_zip64 = True
        else:
            is_zip64 = False
        eocd_offset = tail_offset + eocd_pos
        assert eocd_pos != -1, "Not a valid zip file, could not find EOCD signature"
        if is_zip64:
            # hexdump(tail[eocd_pos:])
            central_dir_offset = struct.unpack('<Q', tail[eocd_pos + 48:eocd_pos + 56])[0]
        else:
            central_dir_offset = struct.unpack('<I', tail[eocd_pos + 16:eocd_pos + 20])[0]
        f.seek(central_dir_offset)
        # print(central_dir_offset)
        # hexdump(out.getvalue()[central_dir_offset:])
        # read the central directory
        # central_dir = io.BytesIO(f.read(eocd_offset - central_dir_offset))
        ret = {}
        while True:
            filename, offset, size = read_central_dir_entry(f)
            if filename is None:
                break
            ret[filename] = offset, size

        return ret

    central_dir = read_central_directory(f)
    # print(central_dir)

    # first_chunk = io.BytesIO(f.read(2048))

    # def parse_pickle(data):

    offset, size = central_dir['archive/data.pkl']
    f.seek(offset)
    filename, data = read_zip_header(f)

    assert filename == 'archive/data.pkl'
    assert len(data) == 0  # data size is stored in central directory, not in the local header

    sizeof = {
        torch.long: 8,
        torch.float: 4,
        torch.half: 2,
        torch.bfloat16: 2,
    }

    def _check_contiguous(shape, stride):
        accum = 1
        for sh, st in reversed(list(zip(shape, stride))):
            if st != accum:
                raise ValueError(f"Expected trivial stride but got {st} for shape {sh}")
            accum *= sh


    # custom pickler to handle persistent id
    class MyUnpickler(pickle.Unpickler):
        def persistent_load(self, data):
            # data example: ('storage', <class 'torch.LongStorage'>, '0', 'cpu', 3)
            _st, storageclass, obj_id, device, numel = data
            obj_id = int(obj_id)
            dummy = storageclass.from_buffer(
                bytes([obj_id] + [0] * (sizeof[storageclass.dtype] - 1)),# + [0] * (sizeof[storageclass] * (numel - 1))),
                "little"
            )
            # dummy[0] = bytes([obj_id])
            return dummy
    #         # return torch.storage.from_buffer(
        # intercept call to function torch._utils._rebuild_tensor_v2
        def find_class(self, module, name):
            def _f(
                storage,
                storage_offset,
                size,
                stride,
                requires_grad,
                backward_hooks,
                metadata=None,
            ):
                # print(storage)
                _check_contiguous(size, stride)  # TODO: handle non contiguous tensors
                stride = (0,) * len(size)  # hack to avoid tensor extending into uninitialized data

                return torch._utils._rebuild_tensor_v2(
                    storage,
                    storage_offset,
                    size,
                    stride,
                    requires_grad,
                    backward_hooks,
                    metadata=metadata,
                )

            if module == 'torch._utils' and name == '_rebuild_tensor_v2':
                # print("intercepting call to _rebuild_tensor_v2")
                return _f
            return super().find_class(module, name)

    # load with custom pickler
    rawdata = f.read(size)
    rawdata = io.BytesIO(rawdata)
    rawdata.seek(0)
    pickler = MyUnpickler(rawdata)
    ob = pickler.load()
    # print(ob)

    dummy_tensor = func(ob)
    assert isinstance(dummy_tensor, torch.Tensor), f"func should point to a tensor inside the pt file but instead found {ob}"

    # print(dummy_tensor.storage())
    id = dummy_tensor.flatten()[:2].contiguous()[:1].view(torch.uint8).numpy().tobytes()
    # print(f"tensor id: {id}")
    id = int.from_bytes(id, 'little')

    real_stride = []
    accum = 1
    for sh in reversed(dummy_tensor.shape):
        real_stride.append(accum)
        accum *= sh

    real_stride = tuple(reversed(real_stride))

    dtype = dummy_tensor.dtype
    my_offset = offset + sum(
        my_idx * my_stride
        for my_idx, my_stride in zip(
            readslice,
            real_stride,
        )  # not strict intentionally!
    ) * sizeof[dtype]
    def prod(iterable):
        result = 1
        for x in iterable:
            result *= x
        return result
    my_size = prod(dummy_tensor.shape[len(readslice):]) * sizeof[dtype]

    offset, size = central_dir[f'archive/data/{id}']
    f.seek(offset, io.SEEK_SET)
    read_zip_header(f)
    f.seek(my_offset, io.SEEK_CUR)
    data = f.read(my_size)

    # make tensor from data
    tensor = torch.from_numpy(
        np.frombuffer(data, dtype=str(dtype).split('.')[-1].replace('bfloat', 'float'))
    ).reshape(dummy_tensor.shape[len(readslice):])
    if dtype == torch.bfloat16:
        tensor = tensor.view(torch.bfloat16)
    tensor = tensor.to(dummy_tensor.device)
    return tensor


