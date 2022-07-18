from pyspark.ml.feature import Normalizer, SQLTransformer
from pyspark.ml.feature import BucketedRandomProjectionLSH
import sparknlp.base as sb
import sparknlp.annotator as sa


SENTENCE_SIM_THRESH = .44
APPROX_JOIN_CUTOFF = .5

def get_word_matching_sql(side):
    """Generate the SQL necessary to transform each side. Side \in {'x', 'y'}"""

    word_pair_min_distance_sql = """
         SELECT entry_id,
                version_x,
                version_y,
                sent_idx_x,
                sent_idx_y,
                word_idx_%(side)s,
                MIN(num_words) as num_words_total_list,
                MIN(distance) as min_word_distance
        FROM __THIS__ 
        GROUP BY entry_id,
                version_x,
                version_y,
                sent_idx_x,
                sent_idx_y,
                word_idx_%(side)s
      """ % ({'side': side})

    sentence_pair_min_distance_sql = """
        SELECT entry_id,
               version_x,
               version_y,
               sent_idx_x,
               sent_idx_y,
               (sum_min_word_distance + %(approx_join_cutoff)f * ( num_words_total - num_matched_words )) / num_words_total AS avg_sentence_distance
        FROM (
           SELECT entry_id,
                  version_x,
                  version_y,
                  sent_idx_x,
                  sent_idx_y,
                  SUM(min_word_distance) AS sum_min_word_distance,
                  COUNT(1) AS num_matched_words,
                  MIN(num_words_total_list) AS num_words_total
           FROM __THIS__
                GROUP BY entry_id,
                   version_x,
                   version_y,
                   sent_idx_x,
                   sent_idx_y
          )
      """ % ({'approx_join_cutoff': APPROX_JOIN_CUTOFF})

    sentence_min_sql = """
         SELECT entry_id,
                version_x,
                version_y,
                sent_idx_x,
                sent_idx_y,
                avg_sentence_distance
           FROM (
                    SELECT *, ROW_NUMBER() OVER (
                         PARTITION BY entry_id, 
                                      version_x, 
                                      version_y, 
                                      sent_idx_%(side)s
                         ORDER BY avg_sentence_distance ASC
                ) AS rn FROM __THIS__
        )
         where rn = 1
    """ % ({'side': side})

    threshold_sql = """
         SELECT entry_id,
                version_x,
                version_y,
                sent_idx_%(join_side)s,
                CASE 
                    WHEN (avg_sentence_distance < %(sentence_sim)f ) THEN sent_idx_%(other_side)s
                    ELSE NULL
                END AS sent_idx_%(other_side)s,
                CASE 
                    WHEN (avg_sentence_distance < %(sentence_sim)f ) THEN avg_sentence_distance
                    ELSE NULL
                END AS avg_sentence_distance
            FROM __THIS__
    """ % ({
        'join_side': side,
        'other_side': list({'x', 'y'} - set(side))[0],
        'sentence_sim': SENTENCE_SIM_THRESH
    })

    return word_pair_min_distance_sql, sentence_pair_min_distance_sql, sentence_min_sql, threshold_sql


#####
#
# Pipelines
#
def get_split_sentence_pipeline():
    documenter = (
        sb.DocumentAssembler()
            .setInputCol("summary")
            .setOutputCol("document")
    )

    sentencer = (
        sa.SentenceDetector()
            .setInputCols(["document"])
            .setOutputCol("sentences")
    )

    sent_finisher = (
        sb.Finisher()
            .setInputCols(["sentences"])
    )

    explode_sent = (
        SQLTransformer()
            .setStatement("""
             SELECT entry_id, version, POSEXPLODE(finished_sentences) AS (sent_idx, sentence)
             FROM __THIS__
        """)
    )

    sentence_splitter_pipeline = sb.Pipeline(stages=[
        documenter,
        sentencer,
        sent_finisher,
        explode_sent
    ])

    return sentence_splitter_pipeline


def get_sparknlp_pipeline(env='bb'):
    ####
    #
    # Spark NLP
    #

    documenter = (
        sb.DocumentAssembler()
            .setInputCol("summary")
            .setOutputCol("document")
    )

    sentencer = (
        sa.SentenceDetector()
            .setInputCols(["document"])
            .setOutputCol("sentences")
    )

    tokenizer = (
        sa.Tokenizer()
            .setInputCols(["sentences"])
            .setOutputCol("token")
    )

    if env=='bb':
        word_embeddings = (
            sa.BertEmbeddings
                .load('s3://aspangher/spark-nlp/small_bert_L4_128_en_2.6.0_2.4')
                .setInputCols(["sentences", "token"])
                .setOutputCol("embeddings")
                .setMaxSentenceLength(512)
                .setBatchSize(100)
        )
        # word_embeddings = (
        #     sa.RoBertaEmbeddings
        #         .load('s3://aspangher/spark-nlp/distilroberta_base_en_3.1.0_2.4')
        #         .setInputCols(["sentences", "token"])
        #         .setOutputCol("embeddings")
        #         .setMaxSentenceLength(512)
        #         .setBatchSize(100)
        # )

    else:
        import os
        local_model_file = 'small_bert_L4_128_en_2.6.0_2.4'
        if not os.path.exists(local_model_file):
            raise FileNotFoundError('Upload model file to this directory!')
        word_embeddings = (
            sa.BertEmbeddings
                .load(local_model_file)
                .setInputCols(["sentences", "token"])
                .setOutputCol("embeddings")
                .setMaxSentenceLength(512)
                .setBatchSize(100)
        )

    tok_finisher = (
        sb.Finisher()
            .setInputCols(["token"])
            .setIncludeMetadata(True)
    )

    embeddings_finisher = (
        sb.EmbeddingsFinisher()
            .setInputCols("embeddings")
            .setOutputCols("embeddings_vectors")
            .setOutputAsVector(True)
    )

    sparknlp_processing_pipeline = sb.Pipeline(stages=[
        documenter,
        sentencer,
        tokenizer,
        word_embeddings,
        embeddings_finisher,
        tok_finisher
      ]
    )
    return sparknlp_processing_pipeline


def get_explode_pipeline():
    ###
    #
    # SQL Processing Steps
    #
    zip_tok = (
        SQLTransformer()
            .setStatement("""
             SELECT CAST(entry_id AS int) as entry_id,
                    CAST(version AS int) as version, 
                    ARRAYS_ZIP(finished_token, finished_token_metadata, embeddings_vectors) AS zipped_tokens
             FROM __THIS__
        """)
    )

    explode_tok = (
        SQLTransformer()
            .setStatement("""
             SELECT entry_id, version, POSEXPLODE(zipped_tokens) AS (word_idx, zipped_token)
             FROM __THIS__
        """)
    )

    rename_tok = (
        SQLTransformer()
            .setStatement("""
             SELECT entry_id, 
                     version,
                     CAST(zipped_token.finished_token_metadata._2 AS int) AS sent_idx,
                     COUNT(1) OVER(PARTITION BY entry_id, version, zipped_token.finished_token_metadata._2) as num_words,
                     CAST(word_idx AS int) word_idx,
                     zipped_token.finished_token AS token,
                     zipped_token.embeddings_vectors as word_embedding
             FROM __THIS__
        """)
    )
    explode_pipeline = sb.PipelineModel(stages=[
        zip_tok,
        explode_tok,
        rename_tok,
    ])

    return explode_pipeline


def get_similarity_pipeline():
    vector_normalizer = (
        Normalizer(
            inputCol="word_embedding",
            outputCol="norm_word_embedding",
            p=2.0
        )
    )
    similarity_checker = (
        BucketedRandomProjectionLSH(
            inputCol="norm_word_embedding",
            outputCol="hashes",
            bucketLength=3,
            numHashTables=3
        )
    )

    similarity_pipeline = sb.Pipeline(stages=[
        vector_normalizer,
        similarity_checker
    ])

    return similarity_pipeline


def get_sentence_pipelines():
    ## get top sentences, X, pipeline
    s1x, s2x, s3x, s4x = get_word_matching_sql(side='x')
    #
    get_word_pair_min_distance_x = SQLTransformer().setStatement(s1x)
    get_sentence_min_distance_x = SQLTransformer().setStatement(s2x)
    get_min_sentence_x = SQLTransformer().setStatement(s3x)
    threshold_x = SQLTransformer().setStatement(s4x)

    ## get top sentences, Y, pipeline
    s1y, s2y, s3y, s4y = get_word_matching_sql(side='y')
    #
    get_word_pair_min_distance_y = SQLTransformer().setStatement(s1y)
    get_sentence_min_distance_y = SQLTransformer().setStatement(s2y)
    get_min_sentence_y = SQLTransformer().setStatement(s3y)
    threshold_y = SQLTransformer().setStatement(s4y)


    top_sentence_pipeline_x = sb.PipelineModel(stages=[
        get_word_pair_min_distance_x,
        get_sentence_min_distance_x,
        get_min_sentence_x,
        threshold_x
    ])

    top_sentence_pipeline_y = sb.PipelineModel(stages=[
        get_word_pair_min_distance_y,
        get_sentence_min_distance_y,
        get_min_sentence_y,
        threshold_y
    ])

    return top_sentence_pipeline_x, top_sentence_pipeline_y