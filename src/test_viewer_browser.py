"""Real Chrome checks for the offline explorer. No annotation inputs are read."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "build/viewer_checks"
    output.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True,
            args=["--no-sandbox", "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        errors, network = [], []
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("request", lambda r: network.append(r.url) if r.url.startswith("http") else None)
        page.goto((ROOT / "out/index.html").as_uri())
        page.wait_for_function("window.MBHExplorer && MBHExplorer.scene.renderer")
        page.wait_for_timeout(500)
        assert page.locator("#entity-name").inner_text() == "Karna"
        assert page.locator("#stage canvas").count() == 1
        assert not page.evaluate("MBHExplorer.scene.renderer.getContext().isContextLost()")
        pixels = page.evaluate("""() => {
          const r=MBHExplorer.scene.renderer,gl=r.getContext(),w=gl.drawingBufferWidth,h=gl.drawingBufferHeight;
          const a=new Uint8Array(w*h*4);gl.readPixels(0,0,w,h,gl.RGBA,gl.UNSIGNED_BYTE,a);
          let varied=0;for(let i=0;i<a.length;i+=4)if(Math.abs(a[i]-a[0])+Math.abs(a[i+1]-a[1])+Math.abs(a[i+2]-a[2])>40)varied++;
          return varied;
        }""")
        assert pixels > 1500, pixels
        initial = page.locator("#stage canvas").screenshot()
        box = page.locator("#stage").bounding_box()
        page.mouse.move(box["x"] + box["width"] * .4, box["y"] + 100)
        page.mouse.down()
        page.mouse.move(box["x"] + box["width"] * .6, box["y"] + 140, steps=12)
        page.mouse.up()
        page.wait_for_timeout(350)
        assert initial != page.locator("#stage canvas").screenshot(), "Orbit did not change the rendered points"
        page.locator("#reset-camera").click()
        page.locator("#dim2").click()
        assert page.locator("#projection-label").inner_text() == "UMAP / 2D"
        assert page.locator("#rotate").is_disabled()
        page.locator("#dim3").click()
        page.locator("#rotate").click()
        assert page.locator("#rotate").get_attribute("aria-pressed") == "true"
        before = page.locator("#stage canvas").screenshot()
        page.wait_for_timeout(400)
        assert before != page.locator("#stage canvas").screenshot()
        page.locator("#rotate").click()

        def query(text):
            page.locator("#q").fill(text)
            page.locator("#query-form").evaluate("f=>f.requestSubmit()")

        assert "bundled model graph" in page.locator('.review-status').inner_text()
        assert page.locator('#clusters button').count() == 8
        assert 'K-means' in page.locator('#clusters button').first.get_attribute('title')
        query('Bhishma')
        parents = page.locator('.relation-group').filter(has=page.locator('h3', has_text='PARENTAGE'))
        assert 'Shantanu' in parents.inner_text()
        assert 'Krishna' not in parents.inner_text()
        assert 'SLAYING ACCOUNTS' in page.locator('#detail-panel').inner_text()
        assert 'not verified distinct killers' in page.locator('#detail-panel').inner_text()
        page.screenshot(path=str(output / 'bhishma-corrected.png'), full_page=True)
        query('Who killed Bhishma?')
        assert 'SOURCE ACCOUNTS' in page.locator('#answer').inner_text()
        assert 'not verified distinct killers' in page.locator('#answer').inner_text()
        query("Vasava")
        assert "Indra" in page.locator('#entity-description').inner_text()
        assert not page.locator('.relation-group [data-entity="'+str(page.evaluate("MBHExplorer.core.resolve('Sanjaya')"))+'"]').count()
        assert not page.locator('.relation-group [data-entity="'+str(page.evaluate("MBHExplorer.core.resolve('Keshava')"))+'"]').count()
        page.locator('.source-usage summary').click()
        assert "Indra" in page.locator('.source-usage blockquote').inner_text()
        page.screenshot(path=str(output / 'vasava-review.png'), full_page=True)
        query("Who is Karna's mother?")
        assert "kind unspecified" in page.locator('#answer').inner_text()
        assert "Radha" in page.locator('#answer').inner_text()
        query("Karna")
        assert page.locator('#graph-container').evaluate('e=>e.scrollTop') == 0, "Do not scroll family headings out of view"
        assert "Radha" in page.locator('#detail-panel').inner_text()
        assert "Kunti" in page.locator('#detail-panel').inner_text()
        slayer_group = page.locator('.relation-group').filter(has=page.locator('h3', has_text='SLAIN BY'))
        assert slayer_group.locator('.relation-item .entity-link').all_text_contents() == ['Arjuna']
        assert 'Kiriti' in slayer_group.locator('.death-names summary').inner_text()
        page.screenshot(path=str(output / 'karna-corrected.png'), full_page=True)
        assert page.locator("#graph .fan-heading").count() > 1
        all_edges = page.locator("#graph .fan-edge").count()
        assert all_edges > 10, "The grouped fan should not retain the old ten-neighbour cap"
        page.locator('#all-connections').click()
        assert page.locator('#graph .fan-edge').count() > all_edges
        page.locator('#all-connections').click()
        assert page.locator('#graph .fan-edge').count() == all_edges
        page.locator('[data-family="conflict"]').click()
        assert page.locator("#graph .fan-edge").count() < all_edges
        page.locator('[data-family="conflict"]').click()
        assert page.locator("#graph .fan-edge").count() == all_edges
        query('Arjuna')
        for category in ['conflict', 'rule', 'divine', 'telling']:
            page.locator('[data-family="'+category+'"]').click()
        assert 'Kunti' in page.locator('#graph').text_content()
        assert 'Kunti' in page.locator('#detail-panel').inner_text()
        page.screenshot(path=str(output / 'arjuna-kinship.png'), full_page=True)
        for category in ['conflict', 'rule', 'divine', 'telling']:
            page.locator('[data-family="'+category+'"]').click()
        page.locator('button[data-layout="graph"]').click()
        assert not page.locator("#stage").is_visible()
        page.locator('button[data-layout="space"]').click()
        assert not page.locator("#graph-container").is_visible()
        assert page.locator("#stage").is_visible()
        page.locator('button[data-layout="both"]').click()
        query("Who taught Karna?")
        answer = page.locator("#answer").inner_text()
        assert "Bhargava" in answer and "08 Karna / 24" in answer
        assert "not in trained graph" in answer
        assert page.locator("#answer .source-passage blockquote").count() > 0
        page.screenshot(path=str(output / "teaching-answer.png"), full_page=True)
        query("Who is Arjuna's teacher?")
        assert "Drona" in page.locator("#answer").inner_text()
        query("How is Karna related to Arjuna?")
        assert 'children of Kunti' in page.locator('#answer .shared-parentage').inner_text()
        page.screenshot(path=str(output / 'shared-parentage.png'), full_page=True)
        assert page.locator("#answer .path-step").count() > 0
        assert "not a verified claim" in page.locator("#answer").inner_text()
        page.locator("#answer [data-evidence]").first.click()
        assert page.locator("#detail-panel blockquote").count() > 0
        uncited_subject = page.evaluate("() => {const k=MBHExplorer.core.data; return k.entities[k.edges.find(e=>!e[5])[0]];}")
        query(uncited_subject)
        page.locator('#all-connections').click()
        assert "uncited" in page.locator("#graph").text_content()
        query("Who fought Bhishma?")
        assert page.locator("#entity-name").inner_text() == "Bhishma"
        assert page.locator("#results tr").count() > 0
        page.locator("#results [data-evidence]").first.click()
        assert page.locator("#detail-panel blockquote").count() > 0
        assert page.locator('[data-detail="evidence"]').get_attribute("aria-selected") == "true"
        query("Who is Draupadi?")
        assert page.locator("#entity-name").inner_text() == "Droupadi"
        assert "Corpus spelling" in page.locator("#message").inner_text()
        query("who are Bhishma's enemies")
        assert "does not establish" in page.locator("#result-note").inner_text()
        query("Who killed Abhimanyu?")
        assert "contested" in page.locator("#result-note").inner_text()
        query("Duryodhana foughtWith ?")
        page.locator("#tab-predictions").click()
        assert page.locator("#results .predicted").count() == 12
        assert page.locator("#results .uncited").count() == 12
        page.locator("#more-results").click()
        assert page.locator("#results tr").count() == 37
        query("Arjuna - Kunti + Gandhari")
        assert page.locator("#result-kind").inner_text() == "VECTOR ARITHMETIC"
        assert page.locator("#results .predicted").count() == 12
        query("Karna")
        with page.expect_download() as export:
            page.locator("#export-results").click()
        exported = json.loads(Path(export.value.path()).read_text())
        assert exported["below_gate"] and exported["results"]
        page.locator("#model-info").click()
        assert page.locator("#model-dialog").is_visible()
        page.locator("#close-model").click()
        page.screenshot(path=str(output / "desktop.png"), full_page=True)

        # Pointer selection must change the focused entity, not just decorate it.
        page.evaluate("MBHExplorer.scene.fit()")
        page.wait_for_timeout(250)
        coord = page.evaluate("MBHExplorer.scene.pointScreen(MBHExplorer.core.resolve('Duryodhana'))")
        box = page.locator("#stage").bounding_box()
        page.mouse.click(box["x"] + coord["x"], box["y"] + coord["y"])
        assert page.locator("#entity-name").inner_text() != "Karna"

        for width,height in [(390,844),(768,1024),(1920,1080)]:
            page.set_viewport_size({"width":width,"height":height})
            query("Karna")
            page.wait_for_timeout(250)
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), width
            assert not page.evaluate("MBHExplorer.scene.renderer.getContext().isContextLost()")
            assert page.locator("#stage canvas").is_visible()
            assert page.evaluate("""() => {
              const stage=document.getElementById('stage');
              return KG.entities.every((_,i)=>{const p=MBHExplorer.scene.pointScreen(i);
                return p.x>0 && p.x<stage.clientWidth && p.y>0 && p.y<stage.clientHeight;});
            }"""), "Projection is clipped at viewport %d" % width
            assert page.evaluate("""() => [...document.querySelectorAll('.fan-node')].every(g=>{
              const t=g.querySelector('text').getBBox(),r=g.querySelector('rect').getBBox();
              return t.x>=r.x && t.x+t.width<=r.x+r.width && t.y>=r.y && t.y+t.height<=r.y+r.height;
            })"""), "A fan label escapes its node"
            page.screenshot(path=str(output / ("viewport-%d.png" % width)), full_page=True)
        assert not errors, errors
        assert not network, network
        browser.close()
    (output / "report.json").write_text(json.dumps({"passed":True,"canvas_pixels":pixels,
        "viewports":[1440,390,768,1920],"external_requests":network,"page_errors":errors},indent=2))
    print("Offline Chrome checks passed: queries, evidence, export, picking, orbit, 2D/3D, four viewports; %d non-background pixels" % pixels)


if __name__ == "__main__":
    main()
