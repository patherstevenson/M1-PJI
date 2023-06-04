#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
:mod:`universe` module
:author: <https://github.com/salaee>
:date: May 2023

universe Module

:doc: <https://github.com/salaee/pegbis>

"""

import numpy as np


class Universe:
    """
    Create a Universe object to represent a disjoint-set forests using union-by-rank and path compression (sort of).
    """
    def __init__(self, n_elements):
        """
        Create a Universe object to represent a disjoint-set forests using union-by-rank and path compression (sort of).
        
        :param n_elements: number of elements (pixels/vertices)
        :type n_elements: int
        :build: a clean Universe for the given number of elements
        """
        self.num = n_elements
        self.elts = np.empty(shape=(n_elements, 3), dtype=int)
        for i in range(n_elements):
            self.elts[i, 0] = 0  # rank
            self.elts[i, 1] = 1  # size
            self.elts[i, 2] = i  # p

    def size(self, x: int) -> int:
        """
        Returns the size of the component to which the given pixel id belongs

        :param x: the pixel id
        :type x: int

        :return: the component id which the given pixel id belongs
        :rtype: int

        :UC: x >= 0
        """
        return self.elts[x, 1]

    def num_sets(self) -> int:
        """
        Return the number of elements

        :return: number of elements/pixels
        :rtype: int

        :UC: None
        """
        return self.num

    def find(self, x: int) -> int:
        """
        Return the component id for a given pixel id

        :param x: pixel id
        :type x: int

        :return:
        :rtype: int

        :UC: x > 0
        """
        y = int(x)
        while y != self.elts[y, 2]:
            y = self.elts[y, 2]
        self.elts[x, 2] = y
        return y

    def join(self, x: int, y: int) -> None:
        """
        Merge the components of the given pixels x and y,
        the most larger component absorb the other

        :param x: the first pixel
        :param y: the second pixel
        :type x: int
        :type y: int

        :return: None
        :rtype: None

        :UC: x > 0, y > 0
        """
        if self.elts[x, 0] > self.elts[y, 0]:
            self.elts[y, 2] = x
            self.elts[x, 1] += self.elts[y, 1]
        else:
            self.elts[x, 2] = y
            self.elts[y, 1] += self.elts[x, 1]
            if self.elts[x, 0] == self.elts[y, 0]:
                self.elts[y, 0] += 1
        self.num -= 1
