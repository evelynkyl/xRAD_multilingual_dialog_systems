#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Dict

from omegaconf import DictConfig
import parlai.utils.logging as logging
from parlai.core.agents import create_agent
from parlai.core.message import Message
from parlai.core.params import ParlaiParser
from parlai.crowdsourcing.tasks.model_chat.constants import AGENT_1
from parlai.utils.strings import normalize_reply
from parlai.crowdsourcing.tasks.model_chat.utils import (
    detect_language,
    MarianTranslatorFromTargetToEN,
    MarianTranslatorFromENToTarget,
    MarianTranslator,
)

class TranslatorTurkLikeAgent:
    """
    Will act like a Turker but actually contains a bot agent.
    """

    def __init__(self, opt, model_name, model_agent, num_turns, semaphore=None):
        self.opt = opt
        self.model_agent = model_agent
        self.id = AGENT_1
        self.num_turns = num_turns
        self.turn_idx = 0
        self.semaphore = semaphore
        self.worker_id = model_name
        self.hit_id = 'none'
        self.assignment_id = 'none'
        self.some_agent_disconnected = False
        self.hit_is_abandoned = False
        self.hit_is_returned = False
        self.disconnected = False
        self.hit_is_expired = False

    def act(self, timeout=None):
        _ = timeout  # The model doesn't care about the timeout
        if self.semaphore:
            with self.semaphore:
                act_out = self.model_agent.act()
        else:
            act_out = self.model_agent.act()
        act_out = Message(act_out).json_safe_payload()

        if 'dict_lower' in self.opt and not self.opt['dict_lower']:
            # model is cased so we don't want to normalize the reply like below
            final_message_text = act_out['text']
            print("this is act_out[text]: ", act_out['text'])
        else:
            final_message_text = normalize_reply(act_out['text'])

        logging.info("Initializing Translator...")
        translator = MarianTranslator(new_ob["text"]) #final_message_text)

        from_tgt_translator = MarianTranslatorFromTargetToEN(translator)
        translator_from_en = MarianTranslatorFromENToTarget(from_tgt_translator)
        logging.info(f"Translating the generated message: {final_message_text} to target language.")
        final_message_text = translator_from_en.translate_answer(final_message_text)
        
        if isinstance(final_message_text, list):
            final_message_text = "".join(final_message_text)
            
        act_out['text'] = final_message_text
        assert ('episode_done' not in act_out) or (not act_out['episode_done'])
        self.turn_idx += 1
        return {**act_out, 'episode_done': False}

    def observe(self, observation, increment_turn: bool = True):
        """
        Need to protect the observe also with a semaphore for composed models where an
        act() may be called within an observe()
        """
        logging.info(
            f'{self.__class__.__name__}: In observe() before semaphore, self.turn_idx is {self.turn_idx} and observation is {observation}'
        )
        global new_ob
        
        new_ob = copy.deepcopy(observation)
        if self.semaphore:
            with self.semaphore:
                self.model_agent.observe(new_ob)
        else:
            self.model_agent.observe(new_ob)
        
        
        if 'text' not in new_ob:
            logging.warning(
                f'{self.__class__.__name__}: In observe() AFTER semaphore, self.turn_idx: {self.turn_idx}, and observation is missing a "text" field: {new_ob}'
            )
        else:            
            if new_ob["text"] == "Hi":
                pass
            else:
                translator = MarianTranslator(new_ob["text"])

                translator_to_en = MarianTranslatorFromTargetToEN(translator)
                logging.info("Translating the incoming message from target language to English.")
                translated_in_message = translator_to_en.translate_question(new_ob["text"])
            
            logging.info(
                f'{self.__class__.__name__}: In observe() AFTER semaphore, self.turn_idx: {self.turn_idx}, observation["text"]: {translated_in_message}'
            )

        if increment_turn:
            self.turn_idx += 1

    def shutdown(self):
        pass

    def reset(self):
        self.model_agent.reset()

    @staticmethod
    def get_bot_agents(
        args: DictConfig, model_opts: Dict[str, str], no_cuda=False
    ) -> Dict[str, dict]:
        """
        Return shared bot agents.

        Pass in model opts with the `model_opts` arg, where `model_opts` is a dictionary
        whose keys are model names and whose values are strings that specify model
        params (i.e. `--model image_seq2seq`).
        """

        # Set up overrides
        model_overrides = {'model_parallel': args.blueprint.task_model_parallel}
        if no_cuda:
            # If we load many models at once, we have to keep it on CPU
            model_overrides['no_cuda'] = no_cuda
        else:
            logging.warning(
                'WARNING: MTurk task has no_cuda FALSE. Models will run on GPU. Will '
                'not work if loading many models at once.'
            )

        # Convert opt strings to Opt objects
        processed_opts = {}
        for name, opt_string in model_opts.items():
            parser = ParlaiParser(True, True)
            parser.set_params(**model_overrides)
            processed_opts[name] = parser.parse_args(opt_string.split())

        # Load and share all model agents
        logging.info(
            f'Got {len(list(processed_opts.keys()))} models: {processed_opts.keys()}.'
        )
        shared_bot_agents = {}
        for model_name, model_opt in processed_opts.items():
            logging.info('\n\n--------------------------------')
            logging.info(f'model_name: {model_name}, opt_dict: {model_opt}')
            model_agent = create_agent(model_opt, requireModelExists=True)
            shared_bot_agents[model_name] = model_agent.share()
        return shared_bot_agents



class TurkLikeAgent:
    """
    Will act like a Turker but actually contains a bot agent.
    """

    def __init__(self, opt, model_name, model_agent, num_turns, semaphore=None):
        self.opt = opt
        self.model_agent = model_agent
        self.id = AGENT_1
        self.num_turns = num_turns
        self.turn_idx = 0
        self.semaphore = semaphore
        self.worker_id = model_name
        self.hit_id = 'none'
        self.assignment_id = 'none'
        self.some_agent_disconnected = False
        self.hit_is_abandoned = False
        self.hit_is_returned = False
        self.disconnected = False
        self.hit_is_expired = False

    def act(self, timeout=None):
        _ = timeout  # The model doesn't care about the timeout
        if self.semaphore:
            with self.semaphore:
                act_out = self.model_agent.act()
        else:
            act_out = self.model_agent.act()
        act_out = Message(act_out).json_safe_payload()

        if 'dict_lower' in self.opt and not self.opt['dict_lower']:
            # model is cased so we don't want to normalize the reply like below
            final_message_text = act_out['text']
        else:
            final_message_text = normalize_reply(act_out['text'])

        act_out['text'] = final_message_text
        assert ('episode_done' not in act_out) or (not act_out['episode_done'])
        self.turn_idx += 1
        return {**act_out, 'episode_done': False}

    def observe(self, observation, increment_turn: bool = True):
        """
        Need to protect the observe also with a semaphore for composed models where an
        act() may be called within an observe()
        """
        logging.info(
            f'{self.__class__.__name__}: In observe() before semaphore, self.turn_idx is {self.turn_idx} and observation is {observation}'
        )
        new_ob = copy.deepcopy(observation)
        if self.semaphore:
            with self.semaphore:
                self.model_agent.observe(new_ob)
        else:
            self.model_agent.observe(new_ob)
        if 'text' not in new_ob:
            logging.warning(
                f'{self.__class__.__name__}: In observe() AFTER semaphore, self.turn_idx: {self.turn_idx}, and observation is missing a "text" field: {new_ob}'
            )
        else:
            logging.info(
                f'{self.__class__.__name__}: In observe() AFTER semaphore, self.turn_idx: {self.turn_idx}, observation["text"]: {new_ob["text"]}'
            )

        if increment_turn:
            self.turn_idx += 1

    def shutdown(self):
        pass

    def reset(self):
        self.model_agent.reset()

    @staticmethod
    def get_bot_agents(
        args: DictConfig, model_opts: Dict[str, str], no_cuda=True
    ) -> Dict[str, dict]:
        """
        Return shared bot agents.

        Pass in model opts with the `model_opts` arg, where `model_opts` is a dictionary
        whose keys are model names and whose values are strings that specify model
        params (i.e. `--model image_seq2seq`).
        """

        # Set up overrides
        model_overrides = {'model_parallel': args.blueprint.task_model_parallel}
        if no_cuda:
            # If we load many models at once, we have to keep it on CPU
            model_overrides['no_cuda'] = no_cuda
        else:
            logging.warning(
                'WARNING: MTurk task has no_cuda FALSE. Models will run on GPU. Will '
                'not work if loading many models at once.'
            )

        # Convert opt strings to Opt objects
        processed_opts = {}
        for name, opt_string in model_opts.items():
            parser = ParlaiParser(True, True)
            parser.set_params(**model_overrides)
            processed_opts[name] = parser.parse_args(opt_string.split())

        # Load and share all model agents
        logging.info(
            f'Got {len(list(processed_opts.keys()))} models: {processed_opts.keys()}.'
        )
        shared_bot_agents = {}
        for model_name, model_opt in processed_opts.items():
            logging.info('\n\n--------------------------------')
            logging.info(f'model_name: {model_name}, opt_dict: {model_opt}')
            model_agent = create_agent(model_opt, requireModelExists=True)
            shared_bot_agents[model_name] = model_agent.share()
        return shared_bot_agents
