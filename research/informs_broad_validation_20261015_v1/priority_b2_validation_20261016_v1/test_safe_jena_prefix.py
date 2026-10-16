"""Tiny generated arrays and streams; never inspect actual Jena numeric values."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import sys
import unittest
from unittest import mock
import zipfile


def load():
    path=Path(__file__).with_name("safe_jena_prefix.py")
    spec=importlib.util.spec_from_file_location("tested_safe_jena_prefix",path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def npy(header,payload):
    text=(repr(header)+"\n").encode("latin1")
    return b"\x93NUMPY\x01\x00"+struct.pack("<H",len(text))+text+payload


class SafePrefixTests(unittest.TestCase):
    def setUp(self):self.api=load()

    def archive(self,blob):
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,"w",compression=zipfile.ZIP_DEFLATED) as archive:archive.writestr("tiny.npy",blob)
        stream.seek(0);return zipfile.ZipFile(stream)

    def test_default_no_input_no_science_import(self):
        before={name:name in sys.modules for name in ("numpy","torch","data_contract")}
        with mock.patch.object(Path,"open",side_effect=AssertionError("default cannot read")),contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(self.api.main([]),0)
        self.assertEqual(json.loads(out.getvalue())["input_reads"],0)
        self.assertEqual(before,{name:name in sys.modules for name in before})

    def test_C_and_F_rectangular_prefix_same_columns_and_tail_never_decoded(self):
        import numpy as np
        matrix=np.arange(15,dtype=np.float32).reshape(5,3);matrix[2:]+=10000
        for order in ("C","F"):
            with self.subTest(order=order):
                calls=[];decode=self.api.decode_numeric
                def record(blob,descr):calls.append(blob);return decode(blob,descr)
                blob=npy(dict(descr="<f4",fortran_order=order=="F",shape=(5,3)),matrix.tobytes(order=order))
                with self.archive(blob) as archive,mock.patch.object(self.api,"decode_numeric",side_effect=record):actual,evidence=self.api.read_member(archive,"tiny.npy",2)
                self.assertTrue(np.array_equal(actual,matrix[:2]))
                expected=[matrix[:2].tobytes()] if order=="C" else [matrix[:2,i].tobytes() for i in range(3)]
                self.assertEqual(calls,expected);self.assertEqual(evidence["numeric_payload_bytes_converted"],24)
                self.assertEqual(evidence["tail_numeric_values_parsed"],0)
                self.assertEqual(evidence["opaque_payload_bytes_discarded"],24 if order=="F" else 0)

    def test_vector_prefix_and_small_metadata(self):
        import numpy as np
        values=np.arange(7,dtype=np.int64)
        with self.archive(npy(dict(descr="<i8",fortran_order=False,shape=(7,)),values.tobytes())) as archive:
            actual,evidence=self.api.read_member(archive,"tiny.npy",3)
        self.assertTrue(np.array_equal(actual,values[:3]));self.assertEqual(evidence["payload_bytes_left_unread"],32)
        with self.archive(npy(dict(descr="<f4",fortran_order=False,shape=(2,)),struct.pack("<ff",1.,2.))) as archive:
            actual,evidence=self.api.read_member(archive,"tiny.npy")
        self.assertEqual(actual.tolist(),[1.,2.]);self.assertEqual(evidence["numeric_payload_bytes_converted"],8)

    def test_object_structured_dtype_and_invalid_shape_rejected_before_decode(self):
        for descr,shape in (("|O8",(2,)),([("a","<f4")],(2,)),("<f4",(-1,)),("<f4",(2,2,2))):
            with self.subTest(descr=descr),mock.patch.object(self.api,"decode_numeric",side_effect=AssertionError("must reject first")):
                with self.assertRaises(ValueError):self.api.read_header(io.BytesIO(npy(dict(descr=descr,fortran_order=False,shape=shape),b"")))

    def test_truncated_header_prefix_or_wrong_payload_size_rejected(self):
        for blob in (b"\x93NUMPY\x01\x00",b"\x93NUMPY\x01\x00"+struct.pack("<H",30)+b"{}"):
            with self.assertRaises(ValueError):self.api.read_header(io.BytesIO(blob))
        blob=npy(dict(descr="<f4",fortran_order=False,shape=(4,2)),b"\0"*8)
        with self.archive(blob) as archive:
            with self.assertRaises(ValueError):self.api.read_member(archive,"tiny.npy",2)
        with self.assertRaises(ValueError):self.api.read_exact(io.BytesIO(b"x"),2)

    def test_header_expansion_and_metadata_decode_bounds(self):
        oversized=b"\x93NUMPY\x02\x00"+struct.pack("<I",65537)
        with self.assertRaises(ValueError):self.api.read_header(io.BytesIO(oversized))
        with self.assertRaises(ValueError):self.api.read_header(io.BytesIO(npy(dict(descr="<f8",fortran_order=False,shape=(100000000,)),b"")))
        blob=npy(dict(descr="|u1",fortran_order=False,shape=(65537,)),b"x"*65537)
        with self.archive(blob) as archive:
            with self.assertRaises(ValueError):self.api.read_member(archive,"tiny.npy")

    def test_prefix_overrun_and_noninteger_rejected(self):
        blob=npy(dict(descr="<f4",fortran_order=True,shape=(3,2)),b"\0"*24)
        for rows in (4,-1,True,1.5):
            with self.archive(blob) as archive,self.subTest(rows=rows):
                with self.assertRaises(ValueError):self.api.read_member(archive,"tiny.npy",rows)

    def test_inherited_full_identity_is_not_prefix_digest(self):
        import numpy as np
        values=np.arange(6,dtype=np.float32).reshape(3,2)
        self.assertNotEqual(self.api.array_sha(values[:2]),self.api.array_sha(values))
        self.assertEqual(self.api.BOUNDS["T_final"],[16772,20965]);self.assertEqual((self.api.SOURCE_ROWS,self.api.PERMITTED_ROWS,self.api.PREFIX_ROWS),(35135,20965,16772))


if __name__=="__main__":unittest.main(verbosity=2)
