import pandas as pd
import unidecode
import json

def get_low_count_versions(conn, high_sent_count=15, low_sent_count=3):
    """Get DF from `split_sentences` with a low and high sentence cutoff."""

    low_count_versions = pd.read_sql('''
        with c1 as (
            SELECT entry_id, 
                    CAST(version as INT) as version, 
                    COUNT(1) as c from split_sentences 
                GROUP BY entry_id, version
        )
        SELECT entry_id, version from c1
        WHERE c < %s and c > %s
    ''' % (high_sent_count, low_sent_count), con=conn)
    return low_count_versions

def get_join_keys(versions_to_get):
    """Convert a dataframe from `matched_sentences` or `split_sentences` to a list of join-keys necessary."""
    # get join keys
    if isinstance(versions_to_get, pd.DataFrame):
        if 'version_x' in versions_to_get.columns:
            joint_keys = (
                versions_to_get[['entry_id', 'version_x', 'version_y']]
                .set_index('entry_id')
                .unstack()
                .reset_index()
                [['entry_id', 0]]
                .drop_duplicates()
                .apply(lambda x: '%s-%s' % (x['entry_id'], x[0]), axis=1)
            )
        else:
            joint_keys = versions_to_get.pipe(lambda df: df['entry_id'].astype(str) + '-' + df['version'].astype(str))
    else:
        joint_keys = versions_to_get
    joint_keys = "'%s'" % "', '".join(joint_keys.tolist())
    return joint_keys

def get_data_from_sqlite_by_sent_cutoffs(source, conn, high_sent_count=15, low_sent_count=3):
    low_count_versions = get_low_count_versions(conn, high_sent_count, low_sent_count)
    join_keys = get_join_keys(low_count_versions)
    return get_data_from_sqlite_by_sentence_criteria(source, conn, join_keys)


def get_data_from_sqlite_by_sentence_criteria(source, conn, join_keys):
    """
    Fetch DFs from `matched_sentences` table and `split_sentences` table where `version_x` and `version_y` are in a
    list called `join_keys`
        -> the form of the join key is just "<entry_id>-<version>"
    .

    :param source:
    :param conn:
    :return:
    """
    # matched sentences
    matched_sentences = pd.read_sql('''
        WITH c1 as ( 
            SELECT *, 
            entry_id || '-' || version_x as key_x,
            entry_id || '-' || version_y as key_y 
            FROM matched_sentences 
        )
        SELECT *
        FROM c1
        WHERE key_x in (%s) AND key_y in (%s)
        ''' % (join_keys, join_keys)
    , con=conn)

    # get split sentences
    split_sentences = pd.read_sql('''
        with c1 AS (
            SELECT *, entry_id || '-' || CAST(version AS INT) as key FROM split_sentences
        )
        SELECT entry_id, CAST(version AS INT) as version, sent_idx, sentence 
        FROM c1
        WHERE key IN (%s)
    ''' % join_keys, con=conn)
    matched_sentences = matched_sentences.assign(source=source)
    split_sentences = split_sentences.assign(source=source)
    return matched_sentences, split_sentences


def match_sentences(matched_sentences, split_sentences):
    """
    Takes as input a `matched_sentences` DF and a `split_sentences` DF and returns a merged DF that can be
    dumped as output for the app, endpoint `/view_task_match`.
    """

    # get HTML diffs
    doc_arcs = (
        matched_sentences
         .merge(split_sentences, how='outer',
                      right_on=['source', 'entry_id', 'version', 'sent_idx'],
                      left_on=['source', 'entry_id', 'version_x', 'sent_idx_x'] ,
          ).drop(['version', 'sent_idx'], axis=1)
         .merge(split_sentences, how='outer',
                      right_on=['source', 'entry_id', 'version', 'sent_idx'],
                      left_on=['source', 'entry_id', 'version_y', 'sent_idx_y'] ,
          ).drop(['version', 'sent_idx'], axis=1)
    )

    grouped_arcs = (
        matched_sentences
         .groupby(['source', 'entry_id', 'version_x', 'version_y'])
         .apply(lambda df:
            df[['version_x', 'version_y', 'sent_idx_x', 'sent_idx_y',
                'avg_sentence_distance_x', 'avg_sentence_distance_y'
               ]].to_dict(orient='records')
         )
         .to_frame('arcs')
    )

    split_sentences['sentence'] = split_sentences['sentence'].apply(unidecode.unidecode)
    split_sentences['sentence'] = split_sentences['sentence'].str.replace('"', '\'\'')
    split_sentences['sentence'] = split_sentences['sentence'].str.replace('<p>', '').str.replace('</p>', '').str.strip()

    grouped_nodes = (
        split_sentences
         .groupby(['source', 'entry_id', 'version'])
         .apply(lambda df: df[['version', 'sent_idx', 'sentence']]
         .to_dict(orient='records'))
         .to_frame('nodes').reset_index()
    )
    matched_grouped_nodes = (
        grouped_nodes
         .merge(
             grouped_nodes.assign(next_vers=lambda df: df['version'] - 1),
             left_on=['source', 'entry_id', 'version'],
             right_on=['source', 'entry_id', 'next_vers']
         )
         .assign(nodes=lambda df: df['nodes_x'] + df['nodes_y'])
         [['source', 'entry_id', 'version_x', 'version_y', 'nodes']]
         .set_index(['source', 'entry_id', 'version_x', 'version_y'])
    )
    output = pd.concat([matched_grouped_nodes, grouped_arcs], axis=1)
    return output


def dump_output_to_app_readable(output_df, outfile=None):
    output = output_df[['nodes', 'arcs']].to_dict(orient='index')
    output = {str(k): v for k, v in output.items()}
    if outfile is not None:
        with open(outfile, 'w') as f:
            json.dump(output, f)
    else:
        return output