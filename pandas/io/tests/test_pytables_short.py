import nose
import sys
import os
import warnings
import tempfile
import shutil
from contextlib import contextmanager
import itertools

import datetime
import numpy as np

import pandas
import pandas as pd
from pandas import (Series, DataFrame, Panel, MultiIndex, Categorical, bdate_range,
                    date_range, timedelta_range, Index, DatetimeIndex, TimedeltaIndex, isnull)

from pandas.compat import is_platform_windows, PY3
from pandas.io.pytables import _tables, TableIterator
try:
    _tables()
except ImportError as e:
    raise nose.SkipTest(e)


from pandas.io.pytables import (HDFStore, get_store, Term, read_hdf,
                                IncompatibilityWarning, PerformanceWarning,
                                AttributeConflictWarning, DuplicateWarning,
                                PossibleDataLossError, ClosedFileError)
from pandas.io import pytables as pytables
import pandas.util.testing as tm
from pandas.util.testing import (assert_panel4d_equal,
                                 assert_panel_equal,
                                 assert_frame_equal,
                                 assert_series_equal)
from pandas import concat, Timestamp
from pandas import compat
from pandas.compat import range, lrange, u
from pandas.util.testing import assert_produces_warning
from numpy.testing.decorators import slow

try:
    import tables
except ImportError:
    raise nose.SkipTest('no pytables')

from distutils.version import LooseVersion

_default_compressor = LooseVersion(tables.__version__) >= '2.2' \
    and 'blosc' or 'zlib'

_multiprocess_can_split_ = False

# testing on windows/py3 seems to fault
# for using compression
skip_compression = PY3 and is_platform_windows()

# contextmanager to ensure the file cleanup
def safe_remove(path):
    if path is not None:
        try:
            os.remove(path)
        except:
            pass


def safe_close(store):
    try:
        if store is not None:
            store.close()
    except:
        pass


def create_tempfile(path):
    """ create an unopened named temporary file """
    return os.path.join(tempfile.gettempdir(),path)


@contextmanager
def ensure_clean_store(path, mode='a', complevel=None, complib=None,
              fletcher32=False):

    try:

        # put in the temporary path if we don't have one already
        if not len(os.path.dirname(path)):
            path = create_tempfile(path)

        store = HDFStore(path, mode=mode, complevel=complevel,
                         complib=complib, fletcher32=False)
        yield store
    finally:
        safe_close(store)
        if mode == 'w' or mode == 'a':
            safe_remove(path)


@contextmanager
def ensure_copied_store(path, mode='a', complevel=None, complib=None,
              fletcher32=False):
    """Copy a store file for non-destructive write testing"""
    
    try:
        base = os.path.basename(path)
        newpath = create_tempfile(base)

        shutil.copy(path, newpath)
        store = HDFStore(newpath, mode=mode, complevel=complevel,
                         complib=complib, fletcher32=False)
        yield store
    finally:
        safe_close(store)
        if mode == 'w' or mode == 'a':
            safe_remove(newpath)

@contextmanager
def ensure_clean_path(path):
    """
    return essentially a named temporary file that is not opened
    and deleted on existing; if path is a list, then create and
    return list of filenames
    """
    try:
        if isinstance(path, list):
            filenames = [ create_tempfile(p) for p in path ]
            yield filenames
        else:
            filenames = [ create_tempfile(path) ]
            yield filenames[0]
    finally:
        for f in filenames:
            safe_remove(f)


# set these parameters so we don't have file sharing
tables.parameters.MAX_NUMEXPR_THREADS = 1
tables.parameters.MAX_BLOSC_THREADS   = 1
tables.parameters.MAX_THREADS   = 1

def _maybe_remove(store, key):
    """For tests using tables, try removing the table to be sure there is
    no content from previous tests using the same table name."""
    try:
        store.remove(key)
    except:
        pass


def compat_assert_produces_warning(w, f):
    """ don't produce a warning under PY3 """
    if compat.PY3:
        f()
    else:
        with tm.assert_produces_warning(expected_warning=w):
            f()


class Base(tm.TestCase):

    @classmethod
    def setUpClass(cls):
        super(Base, cls).setUpClass()

        # Pytables 3.0.0 deprecates lots of things
        tm.reset_testing_mode()

    @classmethod
    def tearDownClass(cls):
        super(Base, cls).tearDownClass()

        # Pytables 3.0.0 deprecates lots of things
        tm.set_testing_mode()

    def setUp(self):
        warnings.filterwarnings(action='ignore', category=FutureWarning)

        self.path = 'tmp.__%s__.h5' % tm.rands(10)

    def tearDown(self):
        pass


class TestHDFStore(Base, tm.TestCase):


    def test_append(self):

        with ensure_clean_store(self.path) as store:
#             df = tm.makeTimeDataFrame()
#             _maybe_remove(store, 'df1')
#             store.append('df1', df[:10])
#             store.append('df1', df[10:])
#             tm.assert_frame_equal(store['df1'], df)
# 
#             _maybe_remove(store, 'df2')
#             store.put('df2', df[:10], format='table')
#             store.append('df2', df[10:])
#             tm.assert_frame_equal(store['df2'], df)
# 
#             _maybe_remove(store, 'df3')
#             store.append('/df3', df[:10])
#             store.append('/df3', df[10:])
#             tm.assert_frame_equal(store['df3'], df)
# 
#             # this is allowed by almost always don't want to do it
#             with tm.assert_produces_warning(expected_warning=tables.NaturalNameWarning):
#                 _maybe_remove(store, '/df3 foo')
#                 store.append('/df3 foo', df[:10])
#                 store.append('/df3 foo', df[10:])
#                 tm.assert_frame_equal(store['df3 foo'], df)
# 
#             # panel
#             wp = tm.makePanel()
#             _maybe_remove(store, 'wp1')
#             store.append('wp1', wp.ix[:, :10, :])
#             store.append('wp1', wp.ix[:, 10:, :])
#             assert_panel_equal(store['wp1'], wp)
# 
#             # ndim
#             p4d = tm.makePanel4D()
#             _maybe_remove(store, 'p4d')
#             store.append('p4d', p4d.ix[:, :, :10, :])
#             store.append('p4d', p4d.ix[:, :, 10:, :])
#             assert_panel4d_equal(store['p4d'], p4d)
# 
#             # test using axis labels
#             _maybe_remove(store, 'p4d')
#             store.append('p4d', p4d.ix[:, :, :10, :], axes=[
#                     'items', 'major_axis', 'minor_axis'])
#             store.append('p4d', p4d.ix[:, :, 10:, :], axes=[
#                     'items', 'major_axis', 'minor_axis'])
#             assert_panel4d_equal(store['p4d'], p4d)
# 
#             # test using differnt number of items on each axis
#             p4d2 = p4d.copy()
#             p4d2['l4'] = p4d['l1']
#             p4d2['l5'] = p4d['l1']
#             _maybe_remove(store, 'p4d2')
#             store.append(
#                 'p4d2', p4d2, axes=['items', 'major_axis', 'minor_axis'])
#             assert_panel4d_equal(store['p4d2'], p4d2)
# 
#             # test using differt order of items on the non-index axes
#             _maybe_remove(store, 'wp1')
#             wp_append1 = wp.ix[:, :10, :]
#             store.append('wp1', wp_append1)
#             wp_append2 = wp.ix[:, 10:, :].reindex(items=wp.items[::-1])
#             store.append('wp1', wp_append2)
#             assert_panel_equal(store['wp1'], wp)

            # dtype issues - mizxed type in a single object column
            df = DataFrame(data=[[1, 2], [0, 1], [1, 2], [0, 0]])
            df['mixed_column'] = 'testing'
            df.ix[2, 'mixed_column'] = np.nan
            _maybe_remove(store, 'df')
            store.append('df', df)
            tm.assert_frame_equal(store['df'], df)

            # uints - test storage of uints
            uint_data = DataFrame({'u08' : Series(np.random.random_integers(0, high=255, size=5), dtype=np.uint8),
                                   'u16' : Series(np.random.random_integers(0, high=65535, size=5), dtype=np.uint16),
                                   'u32' : Series(np.random.random_integers(0, high=2**30, size=5), dtype=np.uint32),
                                   'u64' : Series([2**58, 2**59, 2**60, 2**61, 2**62], dtype=np.uint64)},
                                  index=np.arange(5))
            _maybe_remove(store, 'uints')
            store.append('uints', uint_data)
            tm.assert_frame_equal(store['uints'], uint_data)

            # uints - test storage of uints in indexable columns
            _maybe_remove(store, 'uints')
            store.append('uints', uint_data, data_columns=['u08','u16','u32']) # 64-bit indices not yet supported
            tm.assert_frame_equal(store['uints'], uint_data)


    def test_append_frame_column_oriented(self):

        with ensure_clean_store(self.path) as store:

            # column oriented
            df = tm.makeTimeDataFrame()
            _maybe_remove(store, 'df1')
            store.append('df1', df.ix[:, :2], axes=['columns'])
            store.append('df1', df.ix[:, 2:])
            tm.assert_frame_equal(store['df1'], df)

            result = store.select('df1', 'columns=A')
            expected = df.reindex(columns=['A'])
            tm.assert_frame_equal(expected, result)

            # selection on the non-indexable
            result = store.select(
                'df1', ('columns=A', Term('index=df.index[0:4]')))
            expected = df.reindex(columns=['A'], index=df.index[0:4])
            tm.assert_frame_equal(expected, result)

            # this isn't supported
            self.assertRaises(TypeError, store.select, 'df1', (
                    'columns=A', Term('index>df.index[4]')))

    def test_column_multiindex(self):
        # GH 4710
        # recreate multi-indexes properly

        index = MultiIndex.from_tuples([('A','a'), ('A','b'), ('B','a'), ('B','b')], names=['first','second'])
        df = DataFrame(np.arange(12).reshape(3,4), columns=index)

        with ensure_clean_store(self.path) as store:

            store.put('df',df)
            tm.assert_frame_equal(store['df'],df,check_index_type=True,check_column_type=True)

            store.put('df1',df,format='table')
            tm.assert_frame_equal(store['df1'],df,check_index_type=True,check_column_type=True)

            self.assertRaises(ValueError, store.put, 'df2',df,format='table',data_columns=['A'])
            self.assertRaises(ValueError, store.put, 'df3',df,format='table',data_columns=True)

        # appending multi-column on existing table (see GH 6167)
        with ensure_clean_store(self.path) as store:
            store.append('df2', df)
            store.append('df2', df)

            tm.assert_frame_equal(store['df2'], concat((df,df)))

        # non_index_axes name
        df = DataFrame(np.arange(12).reshape(3,4), columns=Index(list('ABCD'),name='foo'))

        with ensure_clean_store(self.path) as store:

            store.put('df1',df,format='table')
            tm.assert_frame_equal(store['df1'],df,check_index_type=True,check_column_type=True)

    def test_column_multiindex_mixed(self):
        # GH 4710
        # recreate multi-indexes properly

        index = MultiIndex.from_tuples([('A',1), ('A',2), ('B',1), ('B',2)], names=['first','second'])
        df = DataFrame(np.arange(12).reshape(3,4), columns=index)

        with ensure_clean_store(self.path) as store:

            store.put('df',df)
            tm.assert_frame_equal(store['df'],df,check_index_type=True,check_column_type=True)

            store.put('df1',df,format='table')
            tm.assert_frame_equal(store['df1'],df,check_index_type=True,check_column_type=True)

            self.assertRaises(ValueError, store.put, 'df2',df,format='table',data_columns=['A'])
            self.assertRaises(ValueError, store.put, 'df3',df,format='table',data_columns=True)

        # appending multi-column on existing table (see GH 6167)
        with ensure_clean_store(self.path) as store:
            store.append('df2', df)
            store.append('df2', df)

            tm.assert_frame_equal(store['df2'], concat((df,df)))

        # non_index_axes name
        df = DataFrame(np.arange(12).reshape(3,4), columns=Index(list('ABCD'),name='foo'))

        with ensure_clean_store(self.path) as store:

            store.put('df1',df,format='table')
            tm.assert_frame_equal(store['df1'],df,check_index_type=True,check_column_type=True)

    def test_preserve_timedeltaindex_type(self):
        # GH9635
        # Storing TimedeltaIndexed DataFrames in fixed stores did not preserve
        # the type of the index.
        df = DataFrame(np.random.normal(size=(10,5)))
        df.index = timedelta_range(start='0s',periods=10,freq='1s',name='example')

        with ensure_clean_store(self.path) as store:

            store['df'] = df
            assert_frame_equal(store['df'], df)

    def test_wide_table(self):
                   
        index = pd.date_range(start = Timestamp("2015-11-01 0:00"), freq = "H", periods = 3, tz = None)
        columns = MultiIndex.from_tuples(tuple(itertools.product([x for x in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"], 
                                                                 [x for x in "ABCDEFGHIJKLMNOPQRSTUVWXYZ".lower()],
                                                                 "0123456789")))
        data = np.array([index.asi8+i for i in range(len(columns))])
        df   = DataFrame(data.T, columns=columns, index=index) 

        with ensure_clean_store(self.path) as store:
            store.put('df', df, format='table')
            #HDF5ExtError
            tm.assert_frame_equal(store['df'], df, check_index_type=True, check_column_type=True)

           
    def test_legacy_creation(self):
        #"""test the stores that would be created by generate_legacy_storage_files
        
        import pandas.io.tests.generate_legacy_storage_files as gls 
        data = gls.create_hdf_data()
        with ensure_clean_store(self.path) as store:
            for cat in ['series', 'frame', 'panel']:
                for kind, df in sorted(data[cat].items()):
                    fmt = "fixed" if kind == "period" else "table"
                    key = cat + "/" + kind
                    if key in ["frame/td_column", "frame/td_index"]:
                        _x=1
                    store.put(key, df, format = fmt)
                    df_stored = store.get(key)
                    if cat == "series":
                        tm.assert_series_equal(df_stored, df)
                    elif cat == "frame":
                        tm.assert_frame_equal(df_stored, df)
                    elif cat == "panel":
                        tm.assert_panel_equal(df_stored, df)
                
    def test_legacy_stored(self):
        #"""Test compatibility reading legacy versions"""
        
        import pandas.io.tests.generate_legacy_storage_files as gls
        import glob
        
        data = gls.create_hdf_data()
        
        path, _ = gls.get_storage_path(tm.get_data_path("legacy_hdf"), "hdf", "0.17.1", include_platform=False)
        files = glob.glob(path + "/*.h5")
        
        assert len(files) > 0, "no files to test in {}. Configuration problem?".format(path)
    
        for f in files:
            cat, part = os.path.splitext(os.path.basename(f))[0].split("_", 1)
            try:
                df_stored = read_hdf(f, "df")
            except Exception as e:
                _x=1
                pass
            df_new = data[cat][part]
            if cat == "series":
                tm.assert_series_equal(df_stored, df_new)
            elif cat == "frame":
                tm.assert_frame_equal(df_stored, df_new)
            elif cat == "panel":
                tm.assert_panel_equal(df_stored, df_new)
            else:
                raise ValueError("unrecognized data category: " + cat)


    def test_legacy_stored_append(self):
        #"""Test compatibility working with legacy versions"""
        
        import pandas.io.tests.generate_legacy_storage_files as gls
        import glob
        
        data = gls.create_hdf_data()
        
        path, _ = gls.get_storage_path(tm.get_data_path("legacy_hdf"), "hdf", "0.17.1", include_platform=False)
        files = glob.glob(path + "/*.h5")

        assert len(files) > 0, "no files to test in {}. Configuration problem?".format(path)    
            
        for f in files:
            cat, path = os.path.splitext(os.path.basename(f))[0].split("_", 1)
            with ensure_copied_store(f) as store:
                if cat == "panel":
                    continue # I don't know what a sensible test is for panels
#                 if not store.get_storer("df") or not store.get_storer("df").is_table:
#                     continue                
                if not store.get_storer("df").is_table:
                    continue  
                df_new = data[cat][path]
                store.append("df", df_new, dropna=False)
                df_new2 = df_new.append(df_new)
                #df_new2 = pd.concat([df_new, df_new])
                if cat == "series":
                    tm.assert_series_equal(store["df"], df_new2)
                elif cat == "frame":
                    tm.assert_frame_equal(store["df"], df_new2)
                elif cat == "panel":
                    tm.assert_panel_equal(store["df"], df_new2)
                else:
                    raise ValueError("unrecognized data category: " + cat)


def _test_sort(obj):
    if isinstance(obj, DataFrame):
        return obj.reindex(sorted(obj.index))
    elif isinstance(obj, Panel):
        return obj.reindex(major=sorted(obj.major_axis))
    else:
        raise ValueError('type not supported here')


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs'],
                   exit=False)
    #nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
    #               exit=False)
