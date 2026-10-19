import { afterEach, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { defaultNetworkAuto, defaultChannel, networkAutoConfig, draftFromConfig } from './autoConfig'
import { ChannelFields } from './ChannelFields'
import { ChannelHistory } from './ChannelHistory'
import { CrossoverInsights } from './CrossoverInsights'
import { LiveNetworkControl } from './LiveNetworkControl'
import type { DiscussionController } from './DiscussionConfig'
import type { AutoDecision, DiscussionSession } from './types'

afterEach(cleanup)
it('submits MHz and dB channel parameters without treating MHz as Mbps', () => {
  const result = networkAutoConfig(defaultNetworkAuto,'token')
  expect(result.network_mode).toBe('shannon')
  expect(result.channel).toEqual(defaultChannel)
  expect(result.bandwidth_mbps).toBeUndefined()
  expect(draftFromConfig({ ...result, question:'q',rounds:4,max_new_tokens:24,summarizer:'A',mode:'token',handoff_plan:[] }).channel).toEqual(defaultChannel)
})
it.each([{efficiency:0},{correlation:1},{snr_sigma_db:-1},{seed:1.5},{snr_min_db:10,snr_max_db:0}])('rejects invalid channel %j', invalid => {
  expect(() => networkAutoConfig({...defaultNetworkAuto,channel:{...defaultChannel,...invalid}},'token')).toThrow()
})
it('labels standard deviation and advanced clipping independently', () => {
  const change=vi.fn()
  render(<ChannelFields onChange={change} />)
  fireEvent.change(screen.getByLabelText('基准 SNR μ（dB）'),{target:{value:'-5'}})
  expect(change).toHaveBeenCalledWith({...defaultChannel,snr_mean_db:-5})
  expect(screen.getByLabelText('SNR 标准差 σ（dB）')).toHaveValue(4)
  expect(screen.getByLabelText('SNR 截断下界（dB，可留空）')).toHaveValue(null)
  expect(screen.getByText(/σ 是标准差，不是固定上下界/)).toBeInTheDocument()
})
it('plots stored handoff samples and keeps units, revisions and actual delivery', () => {
  const sample={...defaultChannel,snr_db:0,capacity_mbps:20,effective_rate_mbps:16}
  const session={handoffs:[{sequence:1,executed_mode:'token',append_succeeded:true,network:{channel:sample,config_revision:0}},
    {sequence:2,executed_mode:'kv',executed_kv_strategy:'raw_delta',append_succeeded:false,network:{channel:{...sample,snr_db:1},config_revision:1}}]} as unknown as DiscussionSession
  render(<ChannelHistory session={session} />)
  expect(screen.getByRole('img').getAttribute('aria-label')).toContain('2个已记录样本')
  expect(screen.getByText('TOKEN')).toBeInTheDocument()
  expect(screen.getByText('未交付')).toBeInTheDocument()
  expect(screen.queryByText('KV V0')).not.toBeInTheDocument()
  fireEvent.change(screen.getByLabelText('曲线指标'),{target:{value:'effective_rate_mbps'}})
  expect(screen.getByRole('img').getAttribute('aria-label')).toContain('模拟有效速率 R（Mbps）')
})
it('exposes legitimate crossover exploration with synthetic attribution', () => {
  const d={crossover:{status:'positive_crossover',cost_source:'synthetic_assumption',effective_rate_mbps:10,below:'token',above:'kv:v0'},network:{channel:defaultChannel}} as AutoDecision
  render(<CrossoverInsights decision={d} />)
  expect(screen.getByText(/Synthetic · 理想化假设成本/)).toBeInTheDocument()
  expect(screen.getByText(/R\* = 10.000 Mbps/)).toBeInTheDocument()
  const before=screen.getByText(/SNR\*≈/).textContent
  fireEvent.change(screen.getByLabelText('预分析 B（MHz，仅分析）'),{target:{value:'2'}})
  expect(screen.getByText(/SNR\*≈/).textContent).not.toBe(before)
  expect(d.network!.channel!.bandwidth_mhz).toBe(20)
})
it('does not invent a threshold for missing profile or dominant token', () => {
  const {rerender}=render(<CrossoverInsights decision={{crossover:{status:'unavailable'}} as AutoDecision} />)
  expect(screen.getByText(/缺少两个合法候选/)).toBeInTheDocument()
  expect(screen.queryByLabelText('预分析 B（MHz，仅分析）')).not.toBeInTheDocument()
  rerender(<CrossoverInsights decision={{crossover:{status:'no_positive_crossover',preferred:'token'}} as AutoDecision} />)
  expect(screen.getByText(/不存在正的有限交叉点/)).toBeInTheDocument()
})

it('patches a live Shannon channel without sending an independent Mbps rate', async () => {
  const updateNetwork=vi.fn().mockResolvedValue({config_revision:1,effective_from_handoff_index:2})
  const controller={session:{session_id:'current',status:'running',config:{rounds:4,network_mode:'shannon',channel:defaultChannel},network:{config:{mode:'shannon',channel:defaultChannel,base_latency_ms:0},config_revision:0,next_unfrozen_handoff_index:2}},updateNetwork,recovering:false} as unknown as DiscussionController
  render(<LiveNetworkControl controller={controller} />)
  fireEvent.change(screen.getByLabelText('基准 SNR μ（dB）'),{target:{value:'-5'}})
  fireEvent.change(screen.getByLabelText('信道带宽 B（MHz）'),{target:{value:'5'}})
  fireEvent.click(screen.getByRole('button',{name:'应用模拟网络'}))
  await screen.findByText(/版本 1 · 从下一未冻结 Handoff #2/)
  expect(updateNetwork).toHaveBeenCalledWith('current',{channel:{...defaultChannel,snr_mean_db:-5,bandwidth_mhz:5},base_latency_ms:0})
  expect(screen.queryByLabelText('活动模拟带宽 Mbps')).not.toBeInTheDocument()
})
