#coding = utf-8=
import json
import os
import sys
import time
import io
import numpy as np 
class Calligraphy:
    def __init__(self):
        self.WORD_DIR = './all.json'
        word_strokes = []
        self.words = None
        with io.open(self.WORD_DIR, encoding='utf-8') as f:
            res = f.read()
            self.words = json.loads(res)
    def get_arm_strokes(self, words_strokes):
        arm_strokes = []
        for word_id in range(0, len(words_strokes)):
            word = words_strokes[word_id]
            new_word = []
            for stroke in word:
                stroke_ = stroke.reshape([-1, 2])
                arm_stroke_ = stroke_
                new_word.append(arm_stroke_)
            arm_strokes.append(new_word)
        return arm_strokes

    def get_word_strokes(self, word):
        word_strokes = []
        word_info = self.words[word]
        medians = np.array(word_info['medians'])
        for s in medians:
            word_strokes.append(np.array(s))
        return word_strokes
    def get_words(self, words="千山鸟飞绝"):
        all_word_strokes = []
        for i in range(len(words)):
            word = words[i]
            if word == "X":
                continue
            word_strokes = self.get_word_strokes(word)
            all_word_strokes.append(word_strokes)
        return all_word_strokes
    def get_all_points(self, start_pos, words="千山鸟飞绝", axis="1", heng=False, heng_dis = 120, shu_dis=150):
        words_strokes = self.get_words(words)
        new_words_strokes = []
        if axis == "1":
            h = start_pos[1]
            w = start_pos[0]
            w_id = 0
            for word_id in range(0, len(words)):
                if heng:
                    w = w + heng_dis  ##20220909之前都是150
                else:
                    h = h - shu_dis ##字之间的间距  必须是减。##150 ##200
                if words[word_id] == "X":
                    continue
                word = words_strokes[w_id]
                w_id = w_id + 1
                
                new_word = []
                for stroke in word:
                    new_word.append(stroke//7 + np.array([w, h])) ##前面控制字的大小 ##10 ##8
                new_words_strokes.append(new_word)
        if axis == "-":
            pass
        return new_words_strokes
    
