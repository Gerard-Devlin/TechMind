import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { Marked, Renderer } from '../my-project/docs/assets/vendor/ask-ai/marked.esm.js';
const source=readFileSync(new URL('../my-project/docs/javascripts/ask-ai.js',import.meta.url),'utf8');
const setup=source.slice(source.indexOf('  const markdownRenderer'),source.indexOf('  const arrowIcon'));
const markdown=new Function('Marked','Renderer',setup+';return markdown;')(Marked,Renderer);
test('backslash and dollar math is protected before markdown',()=>{
  const result=markdown.parse(String.raw`公式：\(T=(k+n-1)\Delta t\) 和 \[x_1\] 以及 $x_2$`);
  assert.match(result,/tm-ai__math-inline/);
  assert.match(result,/tm-ai__math-display/);
  assert.match(result,/\\Delta t/);
});
test('code remains code and raw html is dropped',()=>{
  const result=markdown.parse('`$x$`\n\n```tex\n\\(x\\)\n```\n\n<img src=x onerror=alert(1)>\n\n**normal**');
  assert.doesNotMatch(result,/tm-ai__math/);
  assert.doesNotMatch(result,/<img/);
  assert.match(result,/<strong>normal<\/strong>/);
});
