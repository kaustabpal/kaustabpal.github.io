(function () {
  var STORAGE_KEY = 'kp-theme';

  function getStored() {
    try { return localStorage.getItem(STORAGE_KEY); } catch (_) { return null; }
  }

  function setStored(v) {
    try { localStorage.setItem(STORAGE_KEY, v); } catch (_) {}
  }

  function applyTheme(theme) {
    var html = document.documentElement;
    var key = document.getElementById('theme-key');
    var label = document.getElementById('theme-label');
    if (theme === 'light') {
      html.classList.add('theme-light');
      if (key) key.textContent = '●';
      if (label) label.textContent = 'Dark';
    } else {
      html.classList.remove('theme-light');
      if (key) key.textContent = '○';
      if (label) label.textContent = 'Light';
    }
  }

  var stored = getStored();
  var initial = stored || (window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
  applyTheme(initial);

  var btn = document.getElementById('theme-toggle');
  if (btn) {
    btn.addEventListener('click', function () {
      var next = document.documentElement.classList.contains('theme-light') ? 'dark' : 'light';
      setStored(next);
      applyTheme(next);
    });
  }

  document.querySelectorAll('a.js-email').forEach(function (a) {
    var u = a.getAttribute('data-u');
    var d = a.getAttribute('data-d');
    if (u && d) a.setAttribute('href', 'mailto:' + u + '@' + d);
  });
})();
