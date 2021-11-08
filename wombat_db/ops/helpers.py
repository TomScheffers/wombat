import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

def groupify_array(arr):
    # Input: Pyarrow/Numpy array
    # Output:
    #   - 1. Unique values
    #   - 2. Sort index
    #   - 3. Count per unique
    #   - 4. Begin index per unique
    dic, counts = np.unique(arr, return_counts=True)
    sort_idx = np.argsort(arr)
    return dic, counts, sort_idx, [0] + np.cumsum(counts)[:-1].tolist()

def combine_column(table, name):
    return table.column(name).combine_chunks()

def _dictionary_and_indices(column):
    assert isinstance(column, pa.ChunkedArray)

    if not isinstance(column.type, pa.DictionaryType):
        column = pc.dictionary_encode(column, null_encoding_behavior='encode')

    dictionary = column.chunk(0).dictionary
    indices = pa.chunked_array([c.indices for c in column.chunks])

    if indices.null_count != 0:
        # We need nulls to be in the dictionary so that indices can be
        # meaningfully multiplied, so we must round trip through decoded
        column = pc.take(dictionary, indices)
        return _dictionary_and_indices(column)

    return dictionary, indices

def columns_to_array(table, columns):
    columns = ([columns] if isinstance(columns, str) else list(set(columns)))
    combined_indices = None
    for c in columns:
        dictionary, indices = _dictionary_and_indices(table.column(c))
        if combined_indices is None:
            combined_indices = indices
        else:
            combined_indices = pc.add(
                pc.multiply(combined_indices, len(dictionary)),
                indices
            )
    return combined_indices.to_numpy()

def tables_to_arrays(table1, table2, columns):
    columns = ([columns] if isinstance(columns, str) else list(set(columns)))
    combined_indices = None
    len1, len2 = table1.num_rows, table2.num_rows 
    for c in columns:
        arr1, arr2 = combine_column(table1, c), combine_column(table2, c)
        dictionary, indices = _dictionary_and_indices(pa.chunked_array(pa.concat_arrays([arr1, arr2.cast(arr1.type)])))
        if combined_indices is None:
            combined_indices = indices
        else:
            combined_indices = pc.add(
                pc.multiply(combined_indices, len(dictionary)),
                indices
            )
    return combined_indices[:len1].to_numpy(), combined_indices[len1:].to_numpy()

# Old helpers

# Splitting tables by columns
def split_array(arr):
    arr = arr.dictionary_encode()
    ind, dic = arr.indices.to_numpy(zero_copy_only=False), arr.dictionary.to_numpy(zero_copy_only=False)

    if len(dic) < 1000:
        # This method is much faster for small amount of categories, but slower for large ones
        return {v: (ind == i).nonzero()[0] for i, v in enumerate(dic)}
    else:
        idxs = [[] for _ in dic]
        [idxs[v].append(i) for i, v in enumerate(ind)]
        return dict(zip(dic, idxs))

def split(table, columns, group=(), idx=None):
    # idx keeps track of the orginal table index, getting split recurrently
    if not isinstance(idx, np.ndarray):
        idx = np.arange(table.num_rows)
    val_idxs = split_array(combine_column(table, columns[0]))
    if columns[1:]:
        return [s for v, i in val_idxs.items() for s in split(table, columns[1:], group + (v,), idx[i])]
    else:
        return [(group + (v,), i) for v, i in val_idxs.items()]