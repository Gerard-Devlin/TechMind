import test from 'node:test';
import assert from 'node:assert/strict';
import { courseImagePath, sourcePageTitle, trustedDocURL, validatedNavigation, validatedSources, readEvents, requestHistory } from '../my-project/docs/javascripts/ask-ai-core.js';
const base='https://gerard-devlin.github.io/TechMind/';

test('document URLs are confined to canonical docs and reject encoded traversal', () => {
  assert.equal(trustedDocURL('COA/7%20%E3%80%81%E6%8C%87%E4%BB%A4%E6%B5%81%E6%B0%B4%E7%BA%BF/#_4',base),base+'COA/7%20%E3%80%81%E6%8C%87%E4%BB%A4%E6%B5%81%E6%B0%B4%E7%BA%BF/#_4');
  for(const url of ['https://evil.example/TechMind/','//evil.example/TechMind/','javascript:alert(1)','../private','https://gerard-devlin.github.io/Other/','C/%252e%252e/x','C/%2f../x','C/%5csecret','C/?token=secret','https://user:pass@gerard-devlin.github.io/TechMind/C/']) assert.equal(trustedDocURL(url,base),null,url);
});

test('source IDs are bounded, deduplicated and cannot add foreign links', () => {
  assert.deepEqual(validatedSources([{id:1,title:'A',url:'C/'},{id:1,title:'B',url:'D/'},{id:2,url:'https://evil.example/'}],base),[{id:1,title:'A',url:base+'C/',excerpt:''}]);
});

test('course artwork follows path slugs and a build-generated local inventory', () => {
  const inventory={coa:'assets/start/courses/coa.webp',mldl:'assets/start/courses/mldl.webp',newcourse:'assets/start/courses/new-course.png'};
  assert.equal(courseImagePath(base+'COA/chapter/',base,inventory),inventory.coa);
  assert.equal(courseImagePath(base+'ML%26DL/Tool/NumPy/',base,inventory),inventory.mldl);
  assert.equal(courseImagePath(base+'New-Course/chapter/',base,inventory),inventory.newcourse);
  assert.equal(courseImagePath(base+'Unknown/chapter/',base,inventory),'assets/start/fallback/signals-gradient.webp');
  assert.equal(courseImagePath(base+'COA/chapter/',base,{coa:'https://evil.example/image.webp'}),'assets/start/fallback/signals-gradient.webp');
});

test('card headings show the document page, not the whole section breadcrumb', () => {
  assert.equal(sourcePageTitle('1、数字逻辑基础 · 二、逻辑函数化简 · 3、卡诺图法'),'1、数字逻辑基础');
  assert.equal(sourcePageTitle('NumPy'),'NumPy');
  assert.equal(sourcePageTitle(''),'Documentation');
});

test('automatic navigation is limited to an exact retrieved document section', () => {
  const sources=validatedSources([{id:1,title:'NumPy',url:'ML%26DL/Tool/NumPy/#broadcast'}],base);
  assert.equal(validatedNavigation({url:sources[0].url},sources,base),sources[0].url);
  for(const value of [null,{}, {url:'https://evil.example/'},{url:'ML%26DL/Tool/NumPy/#other'}, {url:'C/'},{url:'javascript:alert(1)'},{url:'C/%252e%252e/x'}]) assert.equal(validatedNavigation(value,sources,base),null);
  assert.equal(validatedNavigation({url:sources[0].url},[],base),null);
});

test('SSE reads UTF-8 and CRLF at every byte split', async () => {
  const input='event: delta\r\ndata: {"text":"你好🌍"}\r\n\r\nevent: done\r\ndata: {"cached":false}\r\n\r\n';
  const bytes=new TextEncoder().encode(input);
  const stream=new ReadableStream({start(controller){for(const byte of bytes)controller.enqueue(Uint8Array.of(byte));controller.close();}});
  const events=[];for await(const event of readEvents(stream))events.push(event);
  assert.deepEqual(events,[{event:'delta',data:{text:'你好🌍'}},{event:'done',data:{cached:false}}]);
});

test('malformed and interrupted SSE is rejected', async () => {
  for(const input of ['data: invalid\n\n','data: {"x":1}']){
    const stream=new ReadableStream({start(controller){controller.enqueue(new TextEncoder().encode(input));controller.close();}});
    await assert.rejects(async()=>{for await(const event of readEvents(stream))void event;});
  }
});

test('history is restricted to two completed turns', () => {
  const turns=Array.from({length:5},(_,i)=>({question:`q${i}`,answer:`a${i}`,complete:i!==4}));
  assert.deepEqual(requestHistory(turns),[{role:'user',content:'q2'},{role:'assistant',content:'a2'},{role:'user',content:'q3'},{role:'assistant',content:'a3'}]);
});
