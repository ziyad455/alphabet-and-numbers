(function () {
  'use strict';

  window.createDialogueAudio = function createDialogueAudio(basePath, onState) {
    let current = null;
    let currentIndex = -1;
    let generation = 0;

    function report(state, index = currentIndex) {
      onState(state, index);
    }

    function stop() {
      generation += 1;
      if (current) {
        current.pause();
        current.removeAttribute('src');
        current.load();
        current = null;
      }
      currentIndex = -1;
      report('idle', -1);
    }

    function play(index, onEnded) {
      stop();
      const line = window.BEC_DIALOGUE.lines[index];
      if (!line) return;
      const request = generation;
      const audio = new Audio(basePath + line.audio);
      audio.preload = 'auto';
      current = audio;
      currentIndex = index;
      audio.addEventListener('play', () => {
        if (request === generation) report('playing');
      });
      audio.addEventListener('pause', () => {
        if (request === generation && !audio.ended) report('paused');
      });
      audio.addEventListener('ended', () => {
        if (request !== generation) return;
        report('ended');
        if (onEnded) onEnded();
      });
      audio.addEventListener('error', () => {
        if (request === generation) report('error');
      });
      audio.play().catch(error => {
        if (request === generation && error.name !== 'AbortError') report('error');
      });
    }

    function pause() {
      if (current && !current.paused) current.pause();
    }

    function resume() {
      if (current && current.paused && !current.ended) {
        current.play().catch(error => {
          if (error.name !== 'AbortError') report('error');
        });
      }
    }

    return {
      play, pause, resume, stop,
      get index() { return currentIndex; },
      get paused() { return !current || current.paused; },
      get ended() { return !current || current.ended; },
    };
  };
})();
