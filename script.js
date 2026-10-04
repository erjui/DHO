'use strict';

document.querySelectorAll('[data-copy]').forEach(button => {
  button.addEventListener('click', async () => {
    const text = document.getElementById(button.dataset.copy).textContent;
    try {
      await navigator.clipboard.writeText(text);
      button.textContent = 'Copied';
      document.getElementById('copy-status').textContent = 'BibTeX copied to clipboard.';
    } catch {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(document.getElementById(button.dataset.copy));
      selection.removeAllRanges();
      selection.addRange(range);
      button.textContent = 'Text selected';
      document.getElementById('copy-status').textContent = 'Citation selected. Press Control+C or Command+C to copy.';
    }
    window.setTimeout(() => { button.textContent = 'Copy BibTeX'; }, 2500);
  });
});

const modal = document.querySelector('#figure-dialog');
if (modal && typeof modal.showModal === 'function') {
  document.querySelectorAll('[data-zoom]').forEach(link => {
    link.addEventListener('click', event => {
      event.preventDefault();
      const image = link.querySelector('img');
      modal.querySelector('img').src = image.src;
      modal.querySelector('img').alt = image.alt;
      modal.showModal();
    });
  });
  modal.querySelector('button').addEventListener('click', () => modal.close());
  modal.addEventListener('click', event => { if (event.target === modal) modal.close(); });
}

const guideStates = [
  {title:'Identify the user', dialogue:'The user requests a flight change. Their identity has not been established.', result:'Ask for the user ID before looking up the reservation.', gate:'MUTATION GATE · CLOSED', node:0},
  {title:'Check the request against policy', dialogue:'The user has been identified and the reservation loaded. Eligibility still needs to be checked.', result:'Check fare eligibility and the requested flight before offering the change.', gate:'MUTATION GATE · CLOSED', node:2},
  {title:'Obtain explicit confirmation', dialogue:'The change is eligible and its details have been presented. The user has not yet confirmed.', result:'Ask the user to confirm the complete change summary before proceeding.', gate:'MUTATION GATE · CLOSED', node:3},
  {title:'Authorize the requested change', dialogue:'Identity, reservation details, eligibility, and explicit confirmation are grounded in the conversation.', result:'Recommend the change with the verified arguments, then check the tool result.', gate:'MUTATION GATE · OPEN', node:4}
];
const guardStates = [
  {title:'A required step is missing', dialogue:'The agent found a flight and proposed booking it. The conversation contains no insurance offer or explicit confirmation.', result:'Offer insurance and ask the user to confirm the complete booking summary before calling the booking tool.', gate:'BLOCK + REMEDIATION'},
  {title:'The prerequisites are satisfied', dialogue:'The conversation establishes the required details, records the insurance decision, and includes explicit confirmation.', result:'The proposed booking may proceed once the verifier finds every applicable policy requirement satisfied.', gate:'PASS'}
];
document.querySelectorAll('[data-demo]').forEach(demo => {
  const states = demo.dataset.demo === 'guide' ? guideStates : guardStates;
  demo.querySelectorAll('[data-state]').forEach(button => {
    button.addEventListener('click', () => {
      const state = states[Number(button.dataset.state)];
      demo.querySelectorAll('[data-state]').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
      for (const key of ['title', 'dialogue', 'result', 'gate']) demo.querySelector(`[data-output="${key}"]`).textContent = state[key];
      demo.querySelectorAll('[data-node]').forEach(node => {
        const current = Number(node.dataset.node) === state.node;
        node.classList.toggle('active', current);
        if (current) node.setAttribute('aria-current', 'step'); else node.removeAttribute('aria-current');
      });
    });
  });
});

if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      document.querySelectorAll('.nav-links a').forEach(link => {
        if (link.hash === '#' + entry.target.id) link.setAttribute('aria-current', 'location');
        else link.removeAttribute('aria-current');
      });
    });
  }, { rootMargin: '-15% 0px -60% 0px' });
  document.querySelectorAll('main section[id]').forEach(section => observer.observe(section));
}
