import re
from typing import List, Optional, Dict, Any

from .prompts import *
from .retrieval import (
    search_db,
    find_official_sources,
    create_web_collection,
    search_tmp_rag,
)


def extract_number(string: str) -> Optional[int]:
    match = re.search(r"\b[0-4]\b", string)
    return int(match.group()) if match else None


def get_ragger_choice(clients: Dict[str, Any], question: str, docs: str, error: int = 0) -> int:
    open_client_mistral = clients["openai"]["mistralai/Mixtral-8x7B-Instruct-v0.1"]
    model_mistral = [model.id for model in open_client_mistral.models.list()][0]

    try:
        chat_completion = open_client_mistral.chat.completions.create(
            messages=[{"role": "user", "content": get_prompt_ragger(question, docs)}],
            model=model_mistral,
            temperature=0.2,
            max_tokens=3,
            stream=False,
        )
        answer = int(extract_number(chat_completion.choices[0].message.content))

        return answer
    except Exception:
        error += 1
        if error >= 3:
            return 0
        return get_ragger_choice(clients, question, docs, error)


def get_teller_answer(clients: Dict[str, Any], question: str, context: str, choice: int) -> str:
    open_client_llama = clients["openai"]["meta-llama/Meta-Llama-3-8B-Instruct"]
    model_llama = [model.id for model in open_client_llama.models.list()][0]

    prompt = get_prompt_teller(question, context, choice)
    try:
        chat_completion = open_client_llama.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_llama,
            temperature=0.2,
            stream=False,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"Failed to get teller answer: {e}")


def get_checker_answer(clients: Dict[str, Any], question: str, response: str, refs: str) -> str:
    open_client_llama = clients["openai"]["meta-llama/Meta-Llama-3-8B-Instruct"]
    model_llama = [model.id for model in open_client_llama.models.list()][0]

    try:
        chat_completion = open_client_llama.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": get_prompt_checker(question, response, refs),
                }
            ],
            model=model_llama,
            temperature=0.2,
            stream=False,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"Failed to get checker answer: {e}")


def get_googleizer_answer(clients: Dict[str, Any], question: str) -> str:
    open_client_llama = clients["openai"]["meta-llama/Meta-Llama-3-8B-Instruct"]
    model_llama = [model.id for model in open_client_llama.models.list()][0]

    try:
        chat_completion = open_client_llama.chat.completions.create(
            messages=[{"role": "user", "content": get_prompt_googleizer(question)}],
            model=model_llama,
            temperature=0.2,
            stream=False,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"Failed to get Googleizer answer: {e}")


def get_final_answer(
    clients: Dict[str, Any],
    question: str,
    answers: List[str],
    history: List[Dict[str, Any]],
) -> str:
    open_client_mistral = clients["openai"]["mistralai/Mixtral-8x7B-Instruct-v0.1"]
    model_mistral = [model.id for model in open_client_mistral.models.list()][0]

    print("model_mistral: ", model_mistral)

    prompt = get_prompt_concat_answer(answers, question)
    try:
        chat_completion = open_client_mistral.chat.completions.create(
            messages=history[-2:] + [{"role": "user", "content": prompt}],
            model=model_mistral,
            temperature=0.2,
            max_tokens=1024,
            stream=False,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"Failed to get final answer: {e}")


def get_list_web_sources(question: str) -> str:
    results = find_official_sources(question, n=5)
    results_text = "\n".join(
        [
            f"""• {site['title']} ---- {site['href']}\nExtrait : "{site['body']}"\n"""
            for site in results
        ]
    )
    return f"Voilà ce que j'ai trouvé sur internet parmi les sources officielles de l'État :\n\n{results_text}"


async def teller_multi_stuffs(
    clients: Dict[str, Any], prompts: List[str], history: List[Dict[str, Any]]
) -> List[str]:
    open_client_llama = clients["openai"]["meta-llama/Meta-Llama-3-8B-Instruct"]
    model_llama = [model.id for model in open_client_llama.models.list()][0]

    async def multivac_batch_completions(prompts: List[str]):
        try:
            return open_client_llama.completions.create(
                model=model_llama,
                stream=False,
                max_tokens=400,
                temperature=0.2,
                prompt=prompts,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create batch completions: {e}")

    prompts_list = [
        " ".join([f"<|{x['role']}|>{x['content']}\n" for x in history[-2:]]) + f"{prompt}\n"
        for prompt in prompts
    ]

    results = await multivac_batch_completions(prompts_list)
    return [res.text for res in results.choices]


async def go_pipeline(
    question: str,
    docs: List[str] = [],
    refs: List[str] = [],
    n: int = 0,
    fact: int = 5,
    history: Optional[List[Dict[str, Any]]] = None,
    clients: Dict[str, Any] = {},
) -> (str, str):
    print("Go pipeline params: ", question, docs, refs, n, fact, history)
    if not docs:
        docs, refs = search_db(question, k=25)  # Get docs a first time
        docs_tmp = docs[:fact]
        refs_tmp = refs[:fact]
    else:
        docs_tmp = docs[n * fact : (n * fact) + fact]
        refs_tmp = refs[n * fact : (n * fact) + fact]

    if len(docs) > 0:
        context = "\n-------\n".join(docs_tmp[:fact])
        context_refs = refs_tmp[:fact]  # "\n".join(refs_tmp[:fact])
    else:
        context = ""

    print("docs_tmp: ", docs_tmp)

    if question.lower().strip().startswith("web") or question.lower().strip().startswith(
        "internet"
    ):
        choice = 4  # web search
    else:
        choice = get_ragger_choice(clients, question, context)

    print(question, "----> choice: ", choice)

    if choice in [0, 3] and len(docs) >= 1 and n < 3 and (((n * fact) + fact) < len(docs)):
        n += 1
        return await go_pipeline(
            question,
            docs=docs,
            refs=refs,
            n=n,
            fact=fact,
            history=history,
            clients=clients,
        )
    if choice in [1, 2]:
        prompts = get_prompt_teller_multi(question, docs_tmp, choice)
        answers = await teller_multi_stuffs(clients, prompts, history)
        
        prompt = get_prompt_concat_answer(answers, question)

        # Adding refs
        if choice == 1:
            ref_answer = get_checker_answer(clients, question, prompt, "\n".join(context_refs))
        else:
            ref_answer = "🤖"  # "Je n'ai pas utilisé de sources pour cette réponse."
    if choice == 4 or n == 3:  # too much retry ? go internet
        choice = 4
        google_search = get_googleizer_answer(clients, question)

        print(question, "---->", google_search)

        web_results = find_official_sources(google_search)
        if web_results == []:
            prompt = "Désolé je n'ai rien trouvé à ce sujet, ni dans les documents de l'État, ni sur internet !"
            ref_answer = ""
            return prompt, ref_answer
        create_web_collection(web_results)
        docs_tmp = search_tmp_rag(question)
        prompts = get_prompt_teller_multi(question, docs_tmp, choice)
        answers = await teller_multi_stuffs(clients, prompts, history)
        
        prompt = get_prompt_concat_answer(answers, question)

        ref_answer = "\n".join(
            [
                f"""• :url_start:{site['title']} ---- {site['href']}:url_end:\nExtrait : "{site['body']}"\n"""
                for site in web_results
            ]
        )

    elif choice == 0 or (choice == 3):
        prompt = ""
        ref_answer = ""
    
    return prompt, ref_answer
