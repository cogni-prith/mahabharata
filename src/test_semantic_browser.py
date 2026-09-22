"""Exercise the real localhost index and page, including source and security boundaries."""
import json
from pathlib import Path
import urllib.error
import urllib.request
import time
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
URL='http://127.0.0.1:8765'


def call(path,body=None,headers=None):
    h={'Content-Type':'application/json',**(headers or {})}
    req=urllib.request.Request(URL+path,data=json.dumps(body).encode() if body is not None else None,headers=h)
    try:
        with urllib.request.urlopen(req,timeout=30) as response:
            return response.status,json.load(response)
    except urllib.error.HTTPError as error:
        return error.code,json.load(error)


def main():
    code,result=call('/api/search',{'query':'war'})
    assert code==200
    assert result['nodes'][0]['label'].lower()=='war' and result['nodes'][0]['kind']=='Concept'
    assert any(n['label']=='Kurukshetra' and n['basis']=='event_proposals' for n in result['related_places'])
    assert not any(n['label'] in {'oppressed','who','fighting','Duryodhana'} for n in result['related_places'])
    for body in [{'query':''},{'query':'a'*1001},{'query':'war','kind':'unknown'},{'query':'war','limit':True}]:
        assert call('/api/search',body)[0]==400
    assert call('/api/search',{'query':'war OR " DROP TABLE nodes; --'})[0]==200
    assert call('/api/status',headers={'Host':'evil.example'})[0]==403
    assert call('/api/status',headers={'Origin':'https://evil.example'})[0]==403
    assert call('/%2e%2e/CLAUDE.md')[0]==400
    assert call('/api/node?id=not-found')[0]==404
    assert call('/api/write',{'query':'war'})[0]==404
    assert call('/api/status')[0]==200
    output=ROOT/'build/semantic_checks';output.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,
            args=['--no-sandbox','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'])
        page=browser.new_page(viewport={'width':1440,'height':1080},device_scale_factor=1)
        errors=[];external=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('request',lambda r:external.append(r.url) if r.url.startswith('http') and not r.url.startswith(URL) else None)
        page.goto(URL+'/index.html')
        def wait_state(expression):
            # Playwright wait_for_function uses eval, intentionally blocked by our CSP.
            end=time.monotonic()+30
            while time.monotonic()<end:
                if page.evaluate('() => !!('+expression+')'):return
                page.wait_for_timeout(100)
            raise AssertionError(expression+'; '+str(errors))
        wait_state('window.SemanticExplorer?.scene && SemanticExplorer.state.data')
        page.locator('#semantic-inspector h1').filter(has_text='war').wait_for()
        assert not page.locator('.workspace').is_visible()
        assert page.locator('#semantic-stage canvas').count()==1
        assert 'candidate nodes' in page.locator('#stats').inner_text()
        page.wait_for_timeout(400)
        varied=page.evaluate('''() => {
          const gl=SemanticExplorer.scene.renderer.getContext();
          const a=new Uint8Array(gl.drawingBufferWidth*gl.drawingBufferHeight*4);
          gl.readPixels(0,0,gl.drawingBufferWidth,gl.drawingBufferHeight,gl.RGBA,gl.UNSIGNED_BYTE,a);
          let n=0;for(let i=0;i<a.length;i+=4)if(Math.abs(a[i]-a[0])+Math.abs(a[i+1]-a[1])+Math.abs(a[i+2]-a[2])>40)n++;
          return n;
        }''')
        assert varied>1000,varied
        before=page.locator('#semantic-stage canvas').screenshot()
        page.locator('#semantic-rotate').click();page.wait_for_timeout(400)
        assert before!=page.locator('#semantic-stage canvas').screenshot()
        page.locator('#semantic-rotate').click()
        page.locator('#semantic-2d').click()
        assert page.locator('#semantic-rotate').is_disabled()
        assert page.locator('#semantic-projection-label').inner_text()=='TEXT UMAP / 2D'
        page.locator('#semantic-3d').click()
        page.wait_for_timeout(200)
        assert page.evaluate('''() => {
            const boxes=[...document.querySelectorAll('#semantic-labels span:not([hidden])')].map(e=>e.getBoundingClientRect());
            return boxes.every((a,i)=>boxes.slice(i+1).every(b=>a.right<=b.left||b.right<=a.left||a.bottom<=b.top||b.bottom<=a.top));
        }''')
        page.screenshot(path=str(output/'war-desktop.png'),full_page=True)
        page.locator('#semantic-results button').filter(has_text='Kurukshetra').first.click()
        page.locator('#semantic-inspector h1').filter(has_text='Kurukshetra').wait_for()
        assert 'chapter cited' in page.locator('#semantic-inspector').inner_text()
        page.locator('#semantic-inspector summary').filter(has_text='Event-link proposals').click()
        assert page.locator('#semantic-inspector .sem-badge.proposed').count()>0
        page.locator('#semantic-inspector [data-sp]').first.click()
        page.locator('#semantic-inspector .eyebrow').filter(has_text='SOURCE PASSAGE').wait_for()
        page.locator('#semantic-ontology').click()
        page.locator('#semantic-inspector h1').filter(has_text='Classes & evidence').wait_for()
        assert 'Celestial body' in page.locator('#semantic-inspector').inner_text()
        page.locator('[data-sem-class="Place"]').click()
        wait_state("SemanticExplorer.state.data?.nodes.every(n=>n.kind==='Place')")
        assert int(page.locator('#semantic-visible').inner_text().split()[0].replace(',',''))<7000
        page.locator('[data-sem-class=""]').click()
        page.locator('#q').fill('dharma');page.locator('#query-form').evaluate('f=>f.requestSubmit()')
        wait_state("SemanticExplorer.state.data?.query==='dharma'")
        assert page.evaluate("new Set(SemanticExplorer.state.data.nodes.filter(n=>n.label.toLowerCase()==='dharma').map(n=>n.kind)).size")>=2
        page.locator('#mode-graph').click()
        assert page.locator('.workspace').is_visible()
        page.locator('#q').fill('Who taught Karna?');page.locator('#query-form').evaluate('f=>f.requestSubmit()')
        assert 'Bhargava' in page.locator('#answer').inner_text()
        page.locator('#mode-semantic').click()
        for width,height in [(390,844),(768,1024)]:
            page.set_viewport_size({'width':width,'height':height});page.wait_for_timeout(300)
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
            assert page.locator('#semantic-stage canvas').bounding_box()['width']>250
            page.screenshot(path=str(output/f'semantic-{width}.png'),full_page=True)
        assert not errors,errors
        assert not external,external
        page.locator('#q').fill('');page.locator('#query-form').evaluate('f=>f.requestSubmit()')
        page.locator('#semantic-results .sem-error').wait_for()
        assert 'No current results' in page.locator('#semantic-inspector').inner_text()
        offline=browser.new_page();offline.goto((ROOT/'out/index.html').as_uri())
        offline.locator('#mode-semantic').click()
        assert 'offline' in offline.locator('#semantic-status').inner_text()
        offline.locator('#mode-graph').click();assert offline.locator('.workspace').is_visible()
        browser.close()
    print('Semantic checks passed: real war/Kurukshetra retrieval, ambiguity, HTTP boundaries, citations, Three.js pixels/motion, desktop/mobile, graph fallback')


if __name__=='__main__':main()
