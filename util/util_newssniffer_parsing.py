import spacy
import difflib
import itertools
import re
import json
import string
import pycountry
import pandas as pd

nlp = None
spacy_package = 'en_core_web_sm'
def get_nlp():
    global nlp
    if not nlp:
        try:
            nlp = spacy.load(spacy_package, disable=["tagger" "ner"])
        except:
            import subprocess
            print('downloading spacy...')
            subprocess.run("python3 -m spacy download %s" % spacy_package, shell=True)
            nlp = spacy.load(spacy_package, disable=["tagger" "ner"])
    return nlp


nlp_ner = None
def get_nlp_ner():
    global nlp_ner
    if not nlp_ner:
        nlp_ner = spacy.load(spacy_package, disable=["tagger"])  # just the parser
    return nlp_ner


## filter phrases
to_filter = [
    'Share on WhatsApp',
    'Share on Messenger',
    'Reuse this content',
    'Share on LinkedIn',
    'Share on Pinterest' ,
    'Share on Google+',
    'Listen /',
    '– Politics Weekly',
    'Sorry your browser does not support audio',
    'https://flex.acast.com',
    '|',
    'Share on Facebook',
    'Share on Twitter',
    'Share via Email',
    'Sign up to receive',
    'This article is part of a series',
    'Follow Guardian',
    'Twitter, Facebook and Instagram',
    'UK news news',
    'Click here to upload it',
    'Do you have a photo',
    'Listen /',
    'Email View',
    'Read more Guardian',
    'This series is',
    'Readers can recommend ',
    'UK news news',
    'Join the debate',
    'guardian.letters@theguardian.com',
    'More information',
    'Close',
    'All our journalism is independent',
    'is delivered to thousands of inboxes every weekday',
    'with today’s essential stories',
    'Newsflash:',
    'You can read terms of service here',
    'Guardian rating:',
    'By clicking on an affiliate link',
    'morning briefing news',
    'Analysis:',
    'Good morning, and welcome to our rolling coverage',
    'South and Central Asia news',
    'f you have a direct question',
    'sign up to the',
    'You can read terms of service here.',
    'If you want to attract my attention quickly, it is probably better to use Twitter.',
    'UK news',
]
to_filter = list(map(lambda x: x.lower(), to_filter))
starts_with = [
    'Updated ',
    'Here’s the sign-up',
    '[Read more on',
    '[Here’s the list of',
    '[Follow our live coverage',
    '[',
]
contains = [
    'Want to get this briefing by email',
    'Thank youTo'
]
ends_with = [
    ']',
]
last_line_re = re.compile('Currently monitoring (\d|\,)+ news articles')
version_re = re.compile('Version \d+ of \d+')
## general res
clean_escaped_html = re.compile('&lt;.*?&gt;')
end_comma = re.compile(',$')
stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
             "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
             "they",
             "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those",
             "am",
             "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did",
             "doing",
             "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for",
             "with",
             "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to",
             "from",
             "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
             "there",
             "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some",
             "such",
             "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can", "will", "just", "should",
             "now"]
stopwords_lemmas = list(set(map(lambda x: x.lemma_, get_nlp()(' '.join(stopwords)))))
## lambdas
filter_sents = lambda x: not (
    any(map(lambda y: y in x, contains)) or
    any(map(lambda y: x.startswith(y), starts_with)) or
    any(map(lambda y: x.endswith(y), ends_with))
)

def get_words(s, split_method='spacy'):
    if split_method == 'spacy':
        return list(map(lambda x: x.text, get_nlp()(s)))
    else:
        return s.split()

get_lemmas = lambda s: list(map(lambda x: x.lemma_.lower(), get_nlp()(s)))
filter_stopword_lemmas = lambda word_list: list(filter(lambda x: x not in stopwords_lemmas, word_list))
filter_punct = lambda word_list: list(filter(lambda x: x not in string.punctuation, word_list))


###
# I/O methods
#
def clean_html(page):
    lines = page.split('</p><p>')
    if re.search(last_line_re, lines[-1]) is not None:
        lines = lines[:-1]
    if lines[0].startswith('<p><a href'):
        lines = lines[1:]
    if re.search(version_re, lines[0]) is not None:
        lines = lines[1:]
    output_lines = '</p><p>'.join(lines)
    #     format for output
    if output_lines.startswith('<p>'):
        output_lines = output_lines[len('<p>'):]
    if output_lines.endswith('</p>'):
        output_lines = output_lines[:-len('</p>')]

    output_lines = re.sub(clean_escaped_html, '', output_lines)
    return output_lines


def parse_bad_json_line(line):
    if line.strip() == '[' or line.strip() == ']':
        return
    line = re.sub(end_comma, '', line)
    return json.loads(line)


###
# Split sentences and filter lines
#
def is_dateline(x):
    ## is short enough
    length = len(x.split()) < 6
    # has a country name
    # 1. Does it have an uppercase word?
    has_gpe = any(map(lambda x: x.isupper(), x.split()))
    # 2. Is there a country name?
    if not has_gpe:
        for word in get_words(x):
            try:
                pycountry.countries.search_fuzzy(word)
                has_gpe = True
                break
            except LookupError:
                has_gpe = False
    # 3. Is there a GPE?
    if not has_gpe:
        doc = get_nlp_ner()(x)
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                has_gpe = True
    ##
    if length and has_gpe:
        return True
    else:
        return False


def split_sents(a, perform_filter=True):
    nlp = get_nlp()
    output_sents = []

    # deal with dateline (this can really mess things up...)
    dateline_dashes = ['—', '–']
    for d in dateline_dashes:
        dateline = a.split(d)[0]
        if is_dateline(dateline): ## find the dateline
            ## dateline.
            output_sents.append(dateline.strip())
            ## all other sentences.
            a = d.join(a.split(d)[1:]).strip()
            break

    # get sentences for each paragraph
    pars = a.split('</p><p>')
    for p in pars:
        doc = nlp(p)
        sents = list(map(lambda x: x.text, doc.sents))
        output_sents += sents

    # filter out garbage/repetitive sentences
    if perform_filter:
        output_sents = filter_lines(output_sents)

    # last-minute processing
    output_sents = list(map(lambda x: x.strip(), output_sents))

    # merge dateline in with the first sentence
    if len(output_sents) > 0:
        if is_dateline(output_sents[0]):
            output_sents = ['—'.join(output_sents[:2])] + output_sents[2:]
    return output_sents


def filter_lines(a):
    if isinstance(a, list):
        pars = a
    else:
        pars = a.split('</p>')
    output = []
    for p in pars:
        if not any(map(lambda x: x in p.lower(), to_filter)):
            output.append(p)
    if isinstance(a, list):
        return output
    else:
        return '</p>'.join(output)


###
# Get diffs.
#
def get_sentence_diff(a_old, a_new, filter_common_sents=True, merge_clusters=True, slack=.5):
    ## split sentences
    a_old_sents = split_sents(a_old)
    a_new_sents = split_sents(a_new)
    if filter_common_sents:
        a_old_sents = list(filter(filter_sents, a_old_sents))
        a_new_sents = list(filter(filter_sents, a_new_sents))
    ## group list
    vers_old, vers_new = get_list_diff(a_old_sents, a_new_sents)
    ## fix errors/ align sentences
    if merge_clusters:
        vers_old, vers_new = merge_all_clusters(vers_old, vers_new, slack=slack)
    return vers_old, vers_new


def get_word_diffs(s_old, s_new, split_method='spacy'):
    s_old_words, s_new_words = get_words(s_old, split_method), get_words(s_new, split_method)
    return get_list_diff(s_old_words, s_new_words)


def get_word_diff_ratio(s_old, s_new):
    s_old_words, s_new_words = get_words(s_old), get_words(s_new)
    return difflib.SequenceMatcher(None, s_old_words, s_new_words).ratio()


def get_list_diff(l_old, l_new):
    vars_old = []
    vars_new = []
    diffs = list(difflib.ndiff(l_old, l_new))
    in_question = False
    for idx, item in enumerate(diffs):
        label, text = item[0], item[2:]
        if label == '?':
            continue

        elif label == '-':
            vars_old.append({
                'text': text,
                'tag': '-'
            })
            if (
                    ## if something is removed from the old sentnece, a '?' will be present in the next idx
                    ((idx < len(diffs) - 1) and (diffs[idx + 1][0] == '?'))
                    ## if NOTHING is removed from the old sentence, a '?' might still be present in 2 idxs, unless the next sentence is a - as well.
                 or ((idx < len(diffs) - 2) and (diffs[idx + 2][0] == '?') and diffs[idx + 1][0] != '-')
            ):
                in_question = True
                continue

            ## test if the sentences are substantially similar, but for some reason ndiff marked them as different.
            if (idx < len(diffs) - 1) and (diffs[idx + 1][0] == '+'):
                _, text_new = diffs[idx + 1][0], diffs[idx + 1][2:]
                if get_word_diff_ratio(text, text_new) > .8:
                    in_question = True
                    continue

            vars_new.append({
                'text': '',
                'tag': ' '
            })

        elif label == '+':
            vars_new.append({
                'text': text,
                'tag': '+'
            })
            if in_question:
                in_question = False
            else:
                vars_old.append({
                    'text':'',
                    'tag': ' '
                })

        else:
            vars_old.append({
                'text': text,
                'tag': ' '
            })
            vars_new.append({
                'text': text,
                'tag': ' '
            })

    return vars_old, vars_new


###
def get_changes(old_doc, new_doc):
    new_document = []
    old_document = []
    new_sentences = []
    removed_sentences = []

    same_sentences = []
    changed_sentence_pairs = []

    for s_idx, (s_old, s_new) in enumerate(zip(old_doc, new_doc)):
        ###
        if s_old['text'].strip() != '':
            old_document.append(s_old['text'])
        if s_new['text'].strip() != '':
            new_document.append(s_new['text'])

        ###
        if s_old['tag'] == '-' and s_new['tag'] == '+':
            changed_sentence_pairs.append((s_idx, (s_old['text'], s_new['text'])))

        ###
        if s_old['tag'] == ' ' and s_new['tag'] == '+':
            new_sentences.append(s_new['text'])

        ###
        if s_new['tag'] == ' ' and s_old['tag'] == '-':
            removed_sentences.append(s_old['text'])

        if s_new['tag'] == ' ' and s_old['tag'] == ' ':
            same_sentences.append(s_old['text'])

    return {
        'docs': {'old_doc': old_document,
                 'new_doc': new_document,
                 },
        'sentences': {'added_sents': new_sentences,
                      'removed_sents': removed_sentences,
                      'changed_sent_pairs': changed_sentence_pairs
                      }
    }


###
# Fix/align lines
#
def cluster_edits(vo, vn):
    clustered_edits = []
    current_cluster = []
    for o, n in list(zip(vo, vn)):
        if (o['tag'] in ['+', '-']) or (n['tag'] in ['+', '-']):
            current_cluster.append((o, n))
        ##
        if o['tag'] == ' ' and n['tag'] == ' ':
            if len(current_cluster) > 0:
                clustered_edits.append(current_cluster)
                current_cluster = []
            clustered_edits.append([(o, n)])
    if len(current_cluster) > 0:
        clustered_edits.append(current_cluster)
    return clustered_edits


def lemmatize_sentence(s, cache):
    if isinstance(s, str) and s in cache:
        return cache[s], cache
    if isinstance(s, list):
        s = merge_sents_list(s)
    s_lemmas = get_lemmas(s)
    s_lemmas = filter_stopword_lemmas(s_lemmas)
    s_lemmas = filter_punct(s_lemmas)
    cache[s] = s_lemmas
    return cache[s], cache


def check_subset(s1_lemmas, s2_lemmas, slack=.5):
    """Checks if the second sentence is nearly a subset of the first, with up to `slack` words different."""
    ### get all text (might be a list).
    if len(s2_lemmas) > len(s1_lemmas):
        return False
    if len(s2_lemmas) > 50:
        return False
    ### check match.
    matches = sum(map(lambda word: word in s1_lemmas, s2_lemmas))
    return matches >= (len(s2_lemmas) * (1 - slack))


def merge_sents(idx_i, idx_j, a, c):
    """Merges two sentences without spacing errors."""
    si_text = c[idx_i][a]['text']
    sj_text = c[idx_j][a]['text']

    if isinstance(si_text, (list, tuple)):
        output_list = list(si_text)
    else:
        output_list = [(idx_i, si_text)]
    if isinstance(sj_text, (list, tuple)):
        output_list += sj_text
    else:
        output_list.append((idx_j, sj_text))
    return output_list


def merge_sents_list(t):
    t = sorted(t, key=lambda x: x[0])
    t = list(map(lambda x: x[1].strip(), t))
    t = ' '.join(t)
    return ' '.join(t.split())


def text_in_interval(c, idx_i, idx_j, version):
    idx_small, idx_large = min([idx_i, idx_j]), max([idx_i, idx_j])
    return any(map(lambda idx: c[idx][version]['text'].strip() != '',  range(idx_small+1, idx_large)))


def swap_text_spots(c, old_spot_idx, new_spot_idx, version):
    ## swap text
    text_old = c[old_spot_idx][version]['text']
    text_new = c[new_spot_idx][version]['text']
    c[new_spot_idx][version]['text'] = text_old
    c[old_spot_idx][version]['text'] = text_new
    ## swap tags
    tag_new = c[new_spot_idx][version]['tag']
    tag_old = c[old_spot_idx][version]['tag']
    c[new_spot_idx][version]['tag'] = tag_old
    c[old_spot_idx][version]['tag'] = tag_new
    return c

import copy
def merge_cluster(c, slack=.5):
    c = list(filter(lambda x: x[0]['text'] != '' or x[1]['text'] != '', c))
    old_c = copy.deepcopy(c)
    r_c = range(len(c))
    keep_going = True
    loop_idx = 0
    cache = {}

    while keep_going:
        for active_version in [0, 1]:
            inactive_version = abs(active_version - 1)
            for idx_i, idx_j in itertools.product(r_c, r_c):
                # [(0, 0), (0, 1), (1, 0), (1, 1)]
                idx_i, idx_j = (idx_i, idx_j) if active_version == 0 else (idx_j, idx_i)
                if (
                        (idx_i != idx_j)
                        and (c[idx_j][active_version]['text'] != '')
                        # and (c[idx_j][inactive_version]['text'] == '')
                        and (c[idx_i][inactive_version]['text'] != '')
                ):

                    # print('active: %s, idx_i: %s, idx_j: %s' % (active_version, idx_i, idx_j))
                    s1_lemmas, cache = lemmatize_sentence(c[idx_i][inactive_version]['text'], cache)
                    s2_lemmas, cache = lemmatize_sentence(c[idx_j][active_version]['text'], cache)
                    if check_subset(s1_lemmas, s2_lemmas, slack=slack):
                        # if there's a match, first check:
                        combined_text_active = merge_sents(idx_i, idx_j, active_version, c)
                        combined_text_inactive = merge_sents(idx_i, idx_j, inactive_version, c)
                        c[idx_j][active_version]['text'] = combined_text_active
                        c[idx_i][active_version]['text'] = ''
                        c[idx_i][inactive_version]['text'] = combined_text_inactive
                        c[idx_j][inactive_version]['text'] = ''
                        # print('FOUND')
                        # print(c)
                        # print('active: %s, idx_i: %s, idx_j: %s' % (active_version, idx_i, idx_j))

                        #    1. if the two idx's are adjacent, then move the active.
                        if abs(idx_i - idx_j) == 1:
                            # print('1.')
                            c = swap_text_spots(c, new_spot_idx=idx_i, old_spot_idx=idx_j, version=active_version)

                        #    2. if there's both >=1 active AND >=1 inactive in between, don't do anything.
                        elif text_in_interval(c, idx_i, idx_j, active_version) and text_in_interval(c, idx_i, idx_j, inactive_version):
                            # print('2.')
                            pass

                        #    3. if there's text in the active version between the two idx's, move the inactive.
                        elif text_in_interval(c, idx_i, idx_j, active_version):
                            # print('3.')
                            c = swap_text_spots(c, new_spot_idx=idx_j, old_spot_idx=idx_i, version=inactive_version)

                        #    4. if there's text in the inactive in between the two idx's, move the active.
                        elif text_in_interval(c, idx_i, idx_j, inactive_version):
                            # print('4.')
                            c = swap_text_spots(c, new_spot_idx=idx_i, old_spot_idx=idx_j, version=active_version)

                        #   5. if there's no text inbetween the idx's in either the active or the inactive, move the active.
                        elif not (
                                text_in_interval(c, idx_i, idx_j, active_version) and
                                text_in_interval(c, idx_i, idx_j, inactive_version)
                        ):
                            # print('5.')
                            c = swap_text_spots(c, new_spot_idx=idx_i, old_spot_idx=idx_j, version=active_version)

                        ## merge list/text
                        for idx, version in itertools.product([idx_i, idx_j], [active_version, inactive_version]):
                            if isinstance(c[idx][version]['text'], list):
                                c[idx][version]['text'] = merge_sents_list(c[idx][version]['text'])

        ## one more merge for safety
        for idx, version in itertools.product(r_c, [active_version, inactive_version]):
            if isinstance(c[idx][version]['text'], list):
                c[idx][version]['text'] = merge_sents_list(c[idx][version]['text'])

        if (c == old_c) or (loop_idx > 10000):
            # print('done, idx: %s' % loop_idx)
            keep_going = False
            loop_idx = 0
        else:
            loop_idx += 1
            # print('one more')
            old_c = copy.deepcopy(c)

    return c


def merge_all_clusters(vo, vn, slack=.5):
    clustered_edits = cluster_edits(vo, vn)
    output_edits = []
    for c in clustered_edits:
        if len(c) == 1:
            c_i = c[0]
            if not (c_i[0]['text'] == '' and c_i[1]['text'] == ''):
                output_edits.append(c_i)
        else:
            c_new = merge_cluster(c, slack=slack)
            for c_i in c_new:
                if not (c_i[0]['text'] == '' and c_i[1]['text'] == ''):
                    output_edits.append(c_i)

    if len(output_edits) == 0:
        return None, None

    return zip(*output_edits)


def get_sentence_diff_stats(article_df, get_sentence_vars=False, output_type='df'):
    """

    :param article_df:
    :param get_sentence_vars:
    :param get_word_diff:
    :param output_type: `df` or `iter`
    :return:
    """
    import pandas as pd
    from tqdm.auto import tqdm
    sample_ids = article_df['entry_id'].unique()
    article_df = article_df.set_index('entry_id')
    all_sentence_stats, all_word_stats = [], []
    ##
    for a_id in tqdm(sample_ids):
        a = article_df.loc[a_id]
        if len(a) == 1:
            if output_type == 'iter':
                yield None, {'a_id': int(a_id), 'status': 'error, only one version'}
            continue

        if output_type == 'iter':
            yield from get_sentence_diff_stats_on_article_gen(a, a_id, get_sentence_vars)
        else:
            sentence_stats, word_stats = get_sentence_diff_stats_on_article(a, a_id, 'df', get_sentence_vars)
            all_sentence_stats.append(sentence_stats)
            all_word_stats.append(word_stats)

    ## output
    output = pd.concat(all_sentence_stats), pd.concat(all_word_stats)
    return output


def get_sentence_diff_stats_on_article(a, get_sentence_vars, ret_type='df'):
    sentence_stats = []
    word_stats = []

    article_gen = get_sentence_diff_stats_on_article_gen(a, get_sentence_vars)
    for sentence_stat_output, word_stat_output in article_gen:
        if sentence_stat_output is not None:
            sentence_stats.append(sentence_stat_output)
            word_stats.append(word_stat_output)

    if ret_type == 'df':
        return pd.DataFrame(sentence_stats), pd.DataFrame(word_stats)
    else:
        return sentence_stats, word_stats

def get_sentence_diff_stats_on_article_gen(a, get_sentence_vars, a_id=None):
    """

    :param a: is a dataframe of a single article, with all it's versions as rows.
    :param output_type:
    :param get_sentence_vars:
    :param a_id:
    :return:
    """
    if a_id is None:
        a_id = a['entry_id'].unique()[0]

    vs = a['version']
    a_by_v = a.set_index('version')

    for v_old, v_new in list(zip(vs[:-1], vs[1:])):
        try:
            vars_old, vars_new = get_sentence_diff(a_by_v.loc[v_old]['summary'], a_by_v.loc[v_new]['summary'])
        except Exception as e:
            print(e)
            vars_old, vars_new = None, None

        if (vars_old is None and vars_new is None):
            yield None, {'a_id': int(a_id), 'version_old': int(v_old), 'version_new': int(v_new),
                         'status': 'error, no sentences.'}
            continue

        else:
            doc_changes = get_changes(vars_old, vars_new)
            sentence_stat_output = {
                'num_added_sents': len(doc_changes['sentences']['added_sents']),
                'len_new_doc': len(doc_changes['docs']['new_doc']),
                'num_removed_sents': len(doc_changes['sentences']['removed_sents']),
                'len_old_doc': len(doc_changes['docs']['old_doc']),
                'num_changed_sents': len(doc_changes['sentences']['changed_sent_pairs']),
                'version_nums': (v_old, v_new),
                'a_id': a_id,
            }
            if get_sentence_vars:
                sentence_stat_output['vars_old'] = vars_old
                sentence_stat_output['vars_new'] = vars_new
            ##

            ## word diff
            for s_idx, sent_pair in doc_changes['sentences']['changed_sent_pairs']:
                s_old, s_new = get_word_diffs(*sent_pair)
                word_stat_output = {
                    'num_removed_words': sum(map(lambda x: x['tag'] == '-', s_old)),
                    'num_added_words': sum(map(lambda x: x['tag'] == '+', s_new)),
                    'len_old_sent': len(list(filter(lambda x: x['text'] != '', s_old))),
                    'len_new_sent': len(list(filter(lambda x: x['text'] != '', s_new))),
                    'version_nums': (v_old, v_new),
                    's_old': s_old,
                    's_new': s_new,
                    'a_id': a_id,
                    's_idx': s_idx
                }


        if len(doc_changes['sentences']['changed_sent_pairs']) > 0:
            yield (sentence_stat_output, word_stat_output)
        else:
            yield (sentence_stat_output, None)


#######################
#
# HTML Methods
#
#
def html_compare_articles(
        vars_old=None, vars_new=None,
        df=None,
        text_old=None, tags_old=None, text_new=None, tags_new=None
):
    ## reformat variables...
    if vars_old is None and vars_new is None:
        if text_old is None and text_new is None and tags_old is None and tags_new is None:
            df = df.sort_values('s_idx')
            text_old, text_new, tags_old, tags_new = df['sent_old'], df['sent_new'], df['tag_old'], df['tag_new']
        vars_old = list(map(lambda x: {'text': x[0], 'tag': x[1]}, zip(text_old, tags_old)))
        vars_new = list(map(lambda x: {'text': x[0], 'tag': x[1]}, zip(text_new, tags_new)))

    html = [
        '<table>',
        '<tr><th>Old Version</th><th>New Version</th></tr>'
    ]

    for s_old, s_new in zip(vars_old, vars_new):
        row = '<tr>'
        s_old_text, s_new_text = s_old['text'], s_new['text']
        ##
        if s_old['tag'] == '-' and s_new['tag'] == '+':
            s_old_text, s_new_text = html_compare_sentences(*get_word_diffs(s_old_text, s_new_text))
        ##
        if s_old['tag'] == '-':
            row += '<td style="background-color:rgba(255,0,0,0.3)">' + s_old_text + '</td>'
        else:
            row += '<td>' + s_old_text + '</td>'
        ##
        if s_new['tag'] == '+':
            row += '<td style="background-color:rgba(0,255,0,0.3)">' + s_new_text + '</td>'
        else:
            row += '<td>' + s_new_text + '</td>'
        row += '</tr>'
        html.append(row)
    html.append('</table>')
    return '\n'.join(html)


import numpy as np
def is_none_or_nan(x):
    if isinstance(x, str):
        return False
    elif isinstance(x, type(None)):
        return True
    elif np.isnan(x):
        return True
    else:
        return False

def compare_sentences_output(old_sent, new_sent, old_tag_front, new_tag_front, old_tag_back=None, new_tag_back=None, split_method='spacy'):
    if new_tag_back is None:
        new_tag_back = old_tag_back

    if old_tag_back is None:
        old_tag_back = new_tag_back

    if isinstance(old_sent, str) and isinstance(new_sent, str):
        old_sent, new_sent = get_word_diffs(old_sent, new_sent, split_method=split_method)

    elif isinstance(old_sent, str) and is_none_or_nan(new_sent):
        return old_sent, None

    elif is_none_or_nan(old_sent) and isinstance(new_sent, str):
        return None, new_sent

    new_output, old_output = [], []
    for w_old, w_new in zip(old_sent, new_sent):
        if w_old['tag'] == '-':
            old_output.append(old_tag_front + w_old['text'] + old_tag_back)
        else:
            old_output.append(w_old['text'])
        if w_new['tag'] == '+':
            new_output.append(new_tag_front + w_new['text'] + new_tag_back)
        else:
            new_output.append(w_new['text'])

    return ' '.join(old_output), ' '.join(new_output)


def html_compare_sentences(old_sent, new_sent, split_method='spacy', return_dict=False):
    old_html, new_html = compare_sentences_output(
        old_sent, new_sent,
        old_tag_front='<span style="background-color:rgba(255,0,0,0.3)">',
        old_tag_back='</span>',
        new_tag_front='<span style="background-color:rgba(0,255,0,0.3)">',
        new_tag_back='</span>',
        split_method=split_method,
    )

    if return_dict:
        return {'sentence_x_html': old_html, 'sentence_y_html': new_html}
    return old_html, new_html

def latex_compare_sentences(old_sent, new_sent, split_method='spacy', return_dict=False):
    old_latex, new_latex = compare_sentences_output(
        old_sent, new_sent,
        old_tag_front='\hlpink{',
        old_tag_back='}',
        new_tag_front='\hlgreen{',
        new_tag_back='}',
        split_method=split_method,
    )

    if return_dict:
        return {'sentence_x_latex': old_latex, 'sentence_y_html': new_latex}
    return old_latex,  new_latex