import { afterEach, expect, it } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { useState } from 'react'
import { defaultNetworkAuto, defaultSynthetic, draftFromConfig, networkAutoConfig, parseSynthetic, syntheticDraft } from './autoConfig'
import { SyntheticCostFields } from './SyntheticCostFields'
import type { DiscussionConfig } from './types'
afterEach(cleanup)
it('sends and restores exact synthetic assumptions independently of Profile JSON', () => {
  const draft = { ...defaultNetworkAuto, idealized: true, autoPolicy: 'performance_experiment' as const, synthetic: { ...syntheticDraft(), c: '.003', f: '7' }, profileText: 'invalid inactive JSON' }
  const config = networkAutoConfig(draft,'auto')
  expect(config.synthetic_assumptions).toEqual({ ...defaultSynthetic, c: .003, f: 7 })
  expect(config.cost_profile).toBeNull()
  expect(draftFromConfig(config as DiscussionConfig).synthetic).toEqual({ ...syntheticDraft(), c: '0.003', f: '7' })
  expect(() => networkAutoConfig({ ...draft, idealized: false, autoPolicy: 'quality_first', profileText: '' },'auto')).toThrow(/Profile/)
})
it.each(['', ' ', '-1', 'NaN', 'Infinity', '1e309', '1000000001', '0x10'])('rejects invalid coefficient %s', value => {
  expect(() => parseSynthetic({ ...syntheticDraft(), b: value })).toThrow(/系数 b/)
})
it('accepts zero and the documented upper bound', () => {
  expect(parseSynthetic({ ...syntheticDraft(), a:'0', f:'1e9' })).toMatchObject({a:0,f:1e9})
})
it('starts collapsed and resets all six controlled fields and validation', () => {
  function Form() { const [value,setValue]=useState(syntheticDraft()); return <SyntheticCostFields value={value} onChange={setValue} /> }
  const {container} = render(<Form />)
  expect(container.querySelector('details')).not.toHaveAttribute('open')
  fireEvent.click(screen.getByText('高级：计算成本假设'))
  for (const k of ['a','b','c','d','e','f']) fireEvent.change(screen.getByLabelText(`假设系数 ${k}`),{target:{value:'8'}})
  fireEvent.change(screen.getByLabelText('假设系数 a'),{target:{value:''}})
  expect(screen.getByRole('alert')).toHaveTextContent('系数 a')
  fireEvent.click(screen.getByRole('button',{name:'恢复默认假设'}))
  expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  for (const k of ['a','b','c','d','e','f'] as const) expect(screen.getByLabelText(`假设系数 ${k}`)).toHaveValue(defaultSynthetic[k])
  expect(screen.getByText(/Synthetic · 假设成本（非实测 Profile）/)).toBeInTheDocument()
})
