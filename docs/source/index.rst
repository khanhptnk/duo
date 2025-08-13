DuoAI's Documentation!
======================

.. image:: images/logo2.png
   :alt: Duo logo
   :align: center
   :width: 30%
   :target: _none

.. raw:: html

    <div style="margin-bottom: 50px;"></div>


Humans have evolved a `dual cognitive system <https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow>`_: a fast, reactive “System 1” and a slow, logical “System 2.” This architecture makes human decision-making robust and efficient. Robustness comes from System 2’s ability to intervene and prevent mistakes that could be made by the impulsive System 1. Efficiency arises because System 2 can delegate routine tasks to System 1, which operates much faster and with greater energy efficiency.

We develop DuoAI (or Duo for short) to provide the necessary experimental infrastructure to build similar cognitive architecture for AI agents. In this release, we aim to motivate solutions for a fundamental problem: *When should each system of a duo take control of the decision-making process?* We call this problem the `Yield-or-Request Control (YRC) <https://arxiv.org/pdf/2502.09583>`_ problem. Our package is simple and extensible—within just a few lines of code, you can train coordination policies on a wide range of environments, using either existing methods or your own.

We believe that two-system agents are the future of AI. And you can start shaping that future today using Duo!

.. toctree::
   :maxdepth: 2
   :hidden:

   Quickstart <quickstart>
   Core concepts <core_concepts/index>
   Tutorials <tutorials/index>
   Algorithms <algorithms/index>
   API reference <autoapi/index>

Getting Started
---------------

- **If you're new to Duo:** Start with the :doc:`Quickstart <quickstart>` guide to get up and running quickly.
- **Explore core ideas:** The :doc:`Core concepts <core_concepts/index>` section introduces Duo’s fundamental abstractions and design philosophy.
- **Try tutorials:** Visit :doc:`Tutorials <tutorials/index>` for step-by-step examples covering common use cases, from custom environments to new algorithms.
- **See what’s included:** The :doc:`Algorithms <algorithms/index>` page lists the standard algorithms provided in Duo and how to use them.
- **Full API documentation:** Dive into :doc:`API reference <autoapi/index>` for details on classes, methods, and configuration options.

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

