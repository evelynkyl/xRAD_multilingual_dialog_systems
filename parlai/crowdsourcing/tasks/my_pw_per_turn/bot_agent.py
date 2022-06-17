#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import parlai.utils.logging as logging
from parlai.crowdsourcing.tasks.model_chat.bot_agent import TurkLikeAgent, TranslatorTurkLikeAgent
from parlai.core.message import Message
from parlai.utils.strings import normalize_reply
from parlai.crowdsourcing.tasks.model_chat.utils import (
    detect_language,
    MarianTranslatorFromTargetToEN,
    MarianTranslatorFromENToTarget,
    MarianTranslator,
)

class PerTurnEvalTurkLikeAgent(TurkLikeAgent):
    """
    Will act like a Turker but actually contains a bot agent.
    """

    def __init__(self, opt, model_name, model_agent, num_turns, semaphore=None):
        super().__init__(opt, model_name, model_agent, num_turns, semaphore)

    def act(self, timeout=None):
        """
        Same as model chat's bot_agent.py except the self_observe function is removed, a
        custom observe is instead written in worlds.py.

        This is so that the two bots can read each others' messages using observe so
        that the conversation history stays the same.
        """

        _ = timeout  # The model doesn't care about the timeout

        if self.semaphore:
            with self.semaphore:
                act_out = self.model_agent.batch_act([self.model_agent.observation])[0]
        else:
            act_out = self.model_agent.batch_act([self.model_agent.observation])[0]
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


class TranslatorPerTurnEvalTurkLikeAgent(TranslatorTurkLikeAgent):
    """
    Will act like a Turker but actually contains a bot agent.
    """
    
    def __init__(self, opt, model_name, model_agent, num_turns, semaphore=None):
        super().__init__(opt, model_name, model_agent, num_turns, semaphore)

    def act(self, timeout=None):
        """
        Same as model chat's bot_agent.py except the self_observe function is removed, a
        custom observe is instead written in worlds.py.

        This is so that the two bots can read each others' messages using observe so
        that the conversation history stays the same.
        """
        import parlai.crowdsourcing.tasks.my_pw_per_turn.worlds
        
        _ = timeout  # The model doesn't care about the timeout

        if self.semaphore:
            with self.semaphore:
                act_out = self.model_agent.batch_act([self.model_agent.observation])[0]
        else:
            act_out = self.model_agent.batch_act([self.model_agent.observation])[0]
        act_out = Message(act_out).json_safe_payload()

        if 'dict_lower' in self.opt and not self.opt['dict_lower']:
            # model is cased so we don't want to normalize the reply like below
            final_message_text = act_out['text']
        else:
            final_message_text = normalize_reply(act_out['text'])

        logging.info("Initializing Translator...")
        translator = MarianTranslator(parlai.crowdsourcing.tasks.my_pw_per_turn.worlds.greeting_in_lang)
        from_tgt_translator = MarianTranslatorFromTargetToEN(translator)
        translator_from_en = MarianTranslatorFromENToTarget(from_tgt_translator)
        logging.info("Initialization complete!")
        logging.info(f"Translating the generated message: {final_message_text} to target language.")
        final_message_text = translator_from_en.translate_answer(final_message_text)
        logging.info(f"The translate_answer is {final_message_text}.")
        
        if isinstance(final_message_text, list):
            print("List detected, converting it to string")
            final_message_text = "".join(final_message_text)


        act_out['text'] = final_message_text
        assert ('episode_done' not in act_out) or (not act_out['episode_done'])
        self.turn_idx += 1
        return {**act_out, 'episode_done': False}
