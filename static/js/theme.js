(function() {
  document.documentElement.classList.remove('no-js');
  var storageKey = 'theme-preference';

  var getColorPreference = function() {
    if (localStorage.getItem(storageKey))
      return localStorage.getItem(storageKey);
    else
      return 'dark'; // Default to dark mode
  };

  var theme = { value: getColorPreference() };

  var reflectPreference = function() {
    document.documentElement.classList.remove('bg-light', 'bg-dark');
    document.documentElement.classList.add('bg-' + theme.value);
    document.querySelector('#theme-toggle')?.setAttribute('aria-label', theme.value);
    
    // Update icon: moon for light mode, sun for dark mode
    var toggleButton = document.querySelector('#theme-toggle');
    if (toggleButton) {
      toggleButton.textContent = theme.value === 'light' ? '🌓' : '🌓';
    }
  };

  reflectPreference();

  window.onload = function() {
    reflectPreference();

    document.querySelector('#theme-toggle')?.addEventListener('click', function() {
      theme.value = theme.value === 'light' ? 'dark' : 'light';
      localStorage.setItem(storageKey, theme.value);
      reflectPreference();
    });
  };

  window
    .matchMedia('(prefers-color-scheme: dark)')
    .addEventListener('change', function({matches:isDark}) {
      theme.value = isDark ? 'dark' : 'light';
      localStorage.setItem(storageKey, theme.value);
      reflectPreference();
    });
})();